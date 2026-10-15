import json,os,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
from summarize import summarize
from rpent.context.network_baseline import run_baseline, run_gated_baseline
from rpent.evaluation.result import write_json_atomic
ROOT=Path(__file__).resolve().parents[2]
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--batch',required=True,type=Path)
parser.add_argument('--mode',required=True,choices=['full','all','image','text'])
parser.add_argument('--gpu',required=True,type=int)
parser.add_argument('--repeat',required=True,type=int)
parser.add_argument('--port',required=True,type=int)
parser.add_argument('--upstream',default='http://127.0.0.1:19170')
parser.add_argument('--label',required=True)
parser.add_argument('--initial-memory',type=Path,default=ROOT.parent/'initial-memory.tar.gz')
parser.add_argument('--rpent-baseline',action='store_true')
parser.add_argument('--baseline-memory',type=Path)
parser.add_argument('--robot',choices=['libero','robocasa'],default='libero')
parser.add_argument('--task-name')
parser.add_argument('--split',default='target')
parser.add_argument('--vla-model-path',type=Path)
parser.add_argument('--suite',default='libero_object_swap')
parser.add_argument('--task',type=int,default=2)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--prompt-profile',choices=['default','compact'],default='default')
parser.add_argument('--planner-timeout-s',type=int,default=900)
parser.add_argument('--rho',type=float,default=.5)
parser.add_argument('--rho-image',type=float)
parser.add_argument('--rho-text',type=float)
parser.add_argument('--rho-action',type=float)
parser.add_argument('--action-ranges',type=Path)
parser.add_argument('--experience-memory',type=Path)
parser.add_argument('--memory-nc',action='store_true')
parser.add_argument('--memory-blocks',type=Path)
parser.add_argument('--rho-memory',type=float,default=.6)
parser.add_argument('--controller-checkpoint',type=Path)
parser.add_argument('--controller-sample',action='store_true')
parser.add_argument('--controller-seed',type=int,default=0)
parser.add_argument('--controller-behavior',choices=["actor", "lower", "lower-moderate", "lower-strong", "lower-strong-safe"],default='actor')
parser.add_argument('--model',required=True)
parser.add_argument('--reasoning',required=True,choices=['none','low','xhigh'])
args_cli=parser.parse_args()
if args_cli.controller_sample and not args_cli.controller_checkpoint:
    parser.error('--controller-sample requires --controller-checkpoint')
if args_cli.controller_checkpoint and args_cli.mode == 'full':
    parser.error('Full mode uses no learned controller')
from rpent.context.memory import load_memory_blocks
experience_source = args_cli.experience_memory or args_cli.memory_blocks
enable_memory_nc = args_cli.memory_nc or args_cli.memory_blocks is not None
if enable_memory_nc and experience_source is None:
    parser.error('--memory-nc requires --experience-memory')
try:
    if experience_source:
        load_memory_blocks(experience_source)
    if args_cli.memory_blocks and args_cli.experience_memory:
        if load_memory_blocks(args_cli.memory_blocks) != load_memory_blocks(args_cli.experience_memory):
            parser.error('prompt and selector must use the same experience library')
except (OSError, ValueError) as error:
    parser.error(str(error))
if args_cli.rpent_baseline and (args_cli.mode != 'full' or args_cli.baseline_memory is None or experience_source is not None):
    parser.error('RPent baseline requires full mode, baseline memory, and no managed experience')
BASE=args_cli.batch.resolve()
SCRIPTS=Path(__file__).resolve().parent
PYTHON=sys.executable if args_cli.robot == 'robocasa' else '/mnt/public/raojiaji/RPent/.venv-libero/bin/python'
if args_cli.robot == 'robocasa' and (not args_cli.task_name or not args_cli.vla_model_path):
    parser.error('RoboCasa requires --task-name and --vla-model-path')
gpu=args_cli.gpu;queue=[(args_cli.label,args_cli.mode,args_cli.rho)]

for label,mode,rho in queue:
    group=BASE/label;work=group/'workspace';out=group/'output';out.mkdir(parents=True,exist_ok=True)
    state=dict(group=label,gpu=gpu,mode=mode,rho=rho,status='initializing',start_time=time.time())
    def status(**kw):
        state.update(kw);(group/'status.json').write_text(json.dumps(state,indent=2))
    proxy=run=None
    try:
        shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('.git','.venv-context','logs','memory','__pycache__','.pytest_cache','.ruff_cache'),dirs_exist_ok=False)
        (work/'memory').mkdir()
        if args_cli.rpent_baseline:
            shutil.copytree(args_cli.baseline_memory.resolve(),work/'memory',dirs_exist_ok=True)
        if args_cli.robot == 'libero':
            subprocess.run(['tar','xzf',str(args_cli.initial_memory.resolve()),'-C',str(work/'memory')],check=True)
        env=os.environ.copy();env.update(RPENT_REPO_ROOT=str(work),PYTHONPATH=str(work),HF_HUB_OFFLINE='1',RPENT_RUN_OUTPUT=str(out),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        env.setdefault('RPENT_CODEX_LIBERO_PROFILE', '1')
        if args_cli.robot == 'robocasa':
            env['RPENT_CODEX_ROBOCASA_PROFILE']='1'
            env['RPENT_MEMORY_IN_PROMPT']='1'
            env['RPENT_PRESERVE_TOOL_HISTORY']='1'
            env['RLDX_INITIAL_STATE_PATH']=str(
                BASE.parent/'initial_states'/
                f'{args_cli.task_name}_{args_cli.split}_s{args_cli.seed}.json'
            )
        if args_cli.rpent_baseline:
            for key in ('RPENT_CODEX_LIBERO_PROFILE','RPENT_CODEX_ROBOCASA_PROFILE','RPENT_MEMORY_IN_PROMPT'):
                env.pop(key,None)
        temp=group/'tmp';temp.mkdir()
        env.update(TMPDIR=str(temp),TMP=str(temp),TEMP=str(temp),PYTHONDONTWRITEBYTECODE='1')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        port=args_cli.port
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
        env['RPENT_CONTEXT_PROXY_URL']=f'http://127.0.0.1:{port}'
        proxy_args=['/mnt/public/raojiaji/RPent/.venv-libero/bin/python','-m','rpent.context.proxy','--port',str(port),'--upstream',args_cli.upstream,'--encoder','/mnt/public/raojiaji/context-encoder','--mode',mode,'--rho',str(rho),'--cameras',('agentview,navview,wrist' if args_cli.robot == 'robocasa' else 'agentview_policy,agentview_high,wrist_high'),'--log',str(out/'selection.jsonl'),'--audit-dir',str(out/'requests')]
        for modality in ('image','text','action'):
            value=getattr(args_cli, 'rho_'+modality)
            if value is not None: proxy_args.extend(['--rho-'+modality,str(value)])
        memory_path=None
        if experience_source is not None:
            memory_path=group/('experience_memory'+experience_source.suffix)
            shutil.copyfile(experience_source.resolve(),memory_path)
        if enable_memory_nc:
            proxy_args.extend(['--memory-blocks',str(memory_path),'--rho-memory',str(args_cli.rho_memory)])
        if args_cli.action_ranges is not None:
            proxy_args.extend(['--action-ranges',str(args_cli.action_ranges.resolve())])
        if args_cli.controller_checkpoint:
            checkpoint=group/'controller.pt'
            shutil.copyfile(args_cli.controller_checkpoint.resolve(),checkpoint)
            proxy_args.extend(['--controller-checkpoint',str(checkpoint.resolve()),
                               '--controller-seed',str(args_cli.controller_seed),
                               '--controller-behavior',args_cli.controller_behavior])
            if args_cli.controller_sample:
                proxy_args.append('--controller-sample')
        proxy_env=dict(env, PYTHONPATH='/mnt/public/raojiaji/RPent-context-deps:'+str(work))
        if mode != "full" or enable_memory_nc:
            proxy_args.append("--warmup-encoder")
        proxy=subprocess.Popen(proxy_args,cwd=work,env=proxy_env,stdout=open(out/'proxy.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(60):
            if proxy.poll() is not None:raise RuntimeError('proxy exited')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:time.sleep(1)
        else:raise TimeoutError('proxy startup')
        args=[PYTHON,str(SCRIPTS/'run_episode.py'),'--robot',args_cli.robot,'--seed',str(args_cli.seed),'--cuda-device',str(gpu),'--planner','codex','--model',args_cli.model,'--reasoning-effort',args_cli.reasoning,'--planner-timeout-s',str(args_cli.planner_timeout_s),'--max-turns','60','--output-dir',str(out)]
        if args_cli.robot == 'robocasa':
            args.extend(['--task-name',args_cli.task_name,'--split',args_cli.split,'--vla-model-path',str(args_cli.vla_model_path.resolve()),'--memory-profile','local','--memory-dir',str(work/'memory')])
        else:
            args.extend(['--libero-type','pro','--suite',args_cli.suite,'--task',str(args_cli.task),'--prompt-profile',args_cli.prompt_profile])
        if args_cli.rpent_baseline:
            args[1:2]=['-m','rpent.cli.main']
        if memory_path is not None:
            args.extend(['--experience-memory',str(memory_path)])
        (group/'config.json').write_text(json.dumps(dict(argv=args,proxy_argv=proxy_args,initial_memory=str(args_cli.initial_memory.resolve()),model=args_cli.model,gpu=gpu,prompt_profile=args_cli.prompt_profile),indent=2))
        status(status="measuring_network_baseline")
        gate_path = os.environ.get("RPENT_LATENCY_GATE")
        if gate_path:
            def record_check(data):
                write_json_atomic(out/f"network-baseline.check_{data['gate_check']:02d}.json", data)
                print(json.dumps(dict(event="network_baseline_check", check=data["gate_check"], gate=data["latency_gate"])), flush=True)
            baseline = run_gated_baseline(args_cli.upstream, args_cli.model, args_cli.reasoning,
                json.loads(Path(gate_path).read_text()), on_check=record_check)
        else:
            baseline = run_baseline(args_cli.upstream, args_cli.model, args_cli.reasoning)
        write_json_atomic(out/"network-baseline.json", baseline)
        print(json.dumps(dict(event="network_baseline",
            link=baseline["link_summary"], fixed_request=baseline["fixed_request_summary"],
            setup_error=baseline.get("setup_error")), ensure_ascii=False), flush=True)
        if gate_path and not baseline["latency_gate"]["accepted"] and os.environ.get("RPENT_NETWORK_RECORD_ONLY") != "1":
            raise RuntimeError("Latency gate blocked task: " + "; ".join(baseline["latency_gate"]["reasons"]))
        if (baseline.get("link_summary", {}).get("successes") != 5
                or baseline.get("fixed_request_summary", {}).get("successes") != 3):
            raise RuntimeError("Pre-episode network baseline failed; task not started")
        run=subprocess.Popen(args,cwd=work,env=env,stdout=open(out/'launcher.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
        status(status='running',pid=run.pid,proxy_pid=proxy.pid)
        deadline=time.time()+args_cli.planner_timeout_s+200
        while run.poll() is None:
            if time.time()>deadline:raise TimeoutError('episode wall limit 1800s')
            try:summarize(out)
            except Exception as e:(out/'report_error.txt').write_text(repr(e))
            time.sleep(10)
        selection_log=out/'selection.jsonl'
        fallback=selection_log.exists() and any('fallback' in json.loads(line) for line in selection_log.read_text().splitlines())
        status(compression_valid=not fallback)
        status(status='invalid_compression' if fallback else 'finished' if run.returncode==0 else 'process_failed',exit_code=run.returncode,end_time=time.time())
    except Exception as e:
        status(status='error',error=repr(e),end_time=time.time())
    finally:
        for child in (run,proxy):
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        try:summarize(out)
        except Exception as e:(out/'report_error.txt').write_text(repr(e))
