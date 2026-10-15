# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy, cached features for visual scheme B."""

import io
from collections import OrderedDict


class Features:
    """Cache embeddings and 64-bit horizontal difference hashes."""

    _SIMILARITY_CACHE_LIMIT = 4096

    def __init__(self, model: str):
        self.model = model
        self._encoder = None
        self._vectors = {}
        self._hashes = {}
        self._similarities = OrderedDict()

    def warmup(self) -> None:
        """Load and exercise the encoder in the process that will serve requests."""
        self.similarity("context encoder warmup", "context encoder warmup")

    def similarity(self, text: str, query: str) -> float:
        """Return nonnegative cosine similarity, zero for missing metadata."""
        if not text or not query:
            return 0.0
        key = (text, query)
        if key in self._similarities:
            self._similarities.move_to_end(key)
            return self._similarities[key]
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model, device="cpu")
        missing = list(
            dict.fromkeys(s for s in (text, query) if s not in self._vectors)
        )
        if missing:
            values = self._encoder.encode(missing, normalize_embeddings=True)
            self._vectors.update(zip(missing, values))
        score = max(0.0, min(1.0, float(self._vectors[text] @ self._vectors[query])))
        if len(self._similarities) >= self._SIMILARITY_CACHE_LIMIT:
            self._similarities.popitem(last=False)
        self._similarities[key] = score
        return score

    def dhash(self, digest: str, data: bytes) -> int:
        """Return a cached hash without changing the model's original image."""
        if digest not in self._hashes:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                pixels = list(
                    image.convert("L")
                    .resize((9, 8), Image.Resampling.LANCZOS)
                    .getdata()
                )
            bits = (
                pixels[r * 9 + c + 1] > pixels[r * 9 + c]
                for r in range(8)
                for c in range(8)
            )
            self._hashes[digest] = sum(int(bit) << i for i, bit in enumerate(bits))
        return self._hashes[digest]
