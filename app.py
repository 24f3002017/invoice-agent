"""A2A Invoice Agent - one file, Flask."""
import os, json, time, uuid, hashlib, threading, re, traceback
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import requests
from flask import Flask, request, Response

BASE_URL = os.environ.get("BASE_URL", "https://invoice-agent-nxrd.onrender.com/a2a/").rstrip("/") + "/"
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
LLM_URL = os.environ.get("LLM_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
LLM_KEY = os.environ.get("LLM_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
MODELS = [m.strip() for m in os.environ.get("LLM_MODELS", LLM_MODEL).split(",") if m.strip()]
DB_FILE = os.environ.get("DB_FILE", "state.json")

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_REC = "application/vnd.ga5.invoice-action-receipts+json"
A2A = "application/a2a+json"
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}

app = Flask(__name__)
LOCK = threading.RLock()
INFLIGHT = {}
STATE = {"tasks": {}, "msgs": {}, "cache": {}}
if os.path.exists(DB_FILE):
    try:
        STATE.update(json.load(open(DB_FILE)))
    except Exception:
        pass


def log(*a):
    print(*a, flush=True)


def save():  # call while holding LOCK
    tmp = DB_FILE + ".tmp"
    json.dump(STATE, open(tmp, "w"))
    os.replace(tmp, DB_FILE)


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reply(obj, code=200):
    return Response(json.dumps(obj), status=code, mimetype=A2A)


def err(code, msg="Request could not be processed.", tag=None):
    return reply({"error": {"code": code, "status": tag or "ERROR", "message": msg}}, code)


def check(post):
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer ") or not h[7:].strip():
        r = err(401, "Authentication required.")
        r.headers["WWW-Authenticate"] = "Bearer"
        return None, r
    if request.headers.get("A2A-Version") != "1.0":
        return None, err(400, "Unsupported A2A version.")
    if post and request.headers.get("Content-Type", "").split(";")[0].strip() != A2A:
        return None, err(415, "Unsupported media type.")
    return sha(h[7:].strip()), None


@app.get("/.well-known/agent-card.json")
def card():
    return reply({
        "name": "Invoice Action Agent", "description": "Reads invoice packages and proposes safe payment actions.",
        "version": "1.0.0", "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [MT_BATCH], "defaultOutputModes": [MT_PROP, MT_REC],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Choose settle, approve, hold, reject duplicate or open exception for each invoice.",
                    "tags": ["invoice", "payments", "approval"]}],
    })


SYSTEM = f"""You are a careful accounts-payable controller. The data is synthetic.
For EACH invoice package choose exactly one action from: {ACTIONS}.
- settle_invoice: valid, reconciled, within autonomous approval limit.
- request_approval: commercially valid but above delegated authority.
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor+invoice number) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amount, vendor, PO, receipt...) and need an exception workflow.
Find the ONE paragraph that actually determines the action. Ignore the cover sheet, archive/old examples,
training decoys, negated statements ("not a duplicate"), and irrelevant action words.
evidenceRefs = EXACTLY the three bracketed references (copy character-for-character, include the brackets if the document has them) from that deciding paragraph.
facts = vendorName, invoiceNumber, amountMinor (integer, smallest currency unit, e.g. rupees*100), currency.
rationale = 60-1500 chars, name the action and quote at least two evidence refs.
Reply with JSON only: {{"decisions":[{{"packageId":"..","action":"..","facts":{{"vendorName":"..","invoiceNumber":"..","amountMinor":0,"currency":".."}},"evidenceRefs":["..","..",".."],"rationale":".."}}]}}"""


def call_llm(pkgs, deadline=None):
    deadline = deadline or (time.time() + 38)
    last = None
    for attempt in range(5):
        left = deadline - time.time()
        if left < 4:
            break
        model = MODELS[attempt % len(MODELS)]
        try:
            r = requests.post(LLM_URL, timeout=min(left, 25),
                              headers={"Authorization": f"Bearer {LLM_KEY}"},
                              json={"model": model, "temperature": 0,
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": canon(pkgs)}]})
        except Exception as ex:
            last = repr(ex)
            log("LLM EXC", last[:200])
            continue
        if r.status_code == 200:
            try:
                txt = r.json()["choices"][0]["message"]["content"]
                txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
                obj = json.loads(txt)
                return obj if isinstance(obj, list) else obj["decisions"]
            except Exception as ex:
                last = "bad json " + repr(ex)
                log("LLM BAD JSON", last[:200])
                continue
        last = r.status_code
        log("LLM HTTP", r.status_code, r.text[:300])
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5)
            continue
        if r.status_code == 404 and len(MODELS) > 1:
            continue
        break
    raise RuntimeError(f"LLM failed: {last}")


def clean(d, pkg):
    """Return a cleaned decision or None."""
    try:
        text = canon(pkg)
        f = dict(d.get("facts") or {})
        for k in ("vendorName", "invoiceNumber", "amountMinor", "currency"):
            if k not in f:
                return None
        try:
            f["amountMinor"] = int(round(float(f["amountMinor"])))
        except Exception:
            return None
        refs = d.get("evidenceRefs")
        if d.get("action") not in ACTIONS or not isinstance(refs, list):
            return None
        refs = [x for x in dict.fromkeys(refs) if isinstance(x, str) and x in text]
        if len(refs) != 3:
            return None
        r = str(d.get("rationale") or "")
        if not (60 <= len(r) <= 1500 and d["action"] in r and sum(x in r for x in refs) >= 2):
            r = (f"Chosen action: {d['action']}. The deciding paragraph is supported by "
                 f"{refs[0]}, {refs[1]} and {refs[2]}; " + r)[:1500]
        return {"action": d["action"], "facts": f, "evidenceRefs": refs, "rationale": r}
    except Exception:
        return None


def decide(pkgs):
    start = time.time()
    deadline = start + 40
    keys = [sha(canon({k: v for k, v in p.items() if k != "packageId"})) for p in pkgs]
    with LOCK:
        todo = [(p, k) for p, k in zip(pkgs, keys) if k not in STATE["cache"]]

    def run(ch):
        try:
            return {d.get("packageId"): d for d in call_llm([p for p, _ in ch], deadline) if isinstance(d, dict)}
        except Exception as ex:
            log("CHUNK FAILED", repr(ex)[:200])
            return {}

    for attempt in range(2):
        if not todo or (attempt and time.time() - start > 25):
            break
        chunks = [todo[i:i + 3] for i in range(0, len(todo), 3)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outs = list(pool.map(run, chunks))
        out = {}
        for o in outs:
            out.update(o)
        still = []
        for p, k in todo:
            c = clean(out.get(p.get("packageId")) or {}, p)
            if c:
                with LOCK:
                    STATE["cache"][k] = c
            else:
                log("INVALID DECISION for", p.get("packageId"), canon(out.get(p.get("packageId")))[:300])
                still.append((p, k))
        todo = still
        with LOCK:
            save()
    if todo:
        raise RuntimeError("model gave invalid decisions")
    with LOCK:
        return [json.loads(json.dumps(STATE["cache"][k])) for k in keys]


def task_resp(tid):
    with LOCK:
        return reply({"task": STATE["tasks"][tid]["task"]})


@app.post(BASE_PATH + "/message:send")
def send():
    p, e = check(True)
    if e:
        return e
    body = request.get_json(silent=True)
    msg = body.get("message") if isinstance(body, dict) else None
    if not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or msg.get("role") != "ROLE_USER":
        return err(400, "Invalid message.")
    h = sha(canon(msg))
    key = f"{p}|{msg['messageId']}"
    with LOCK:
        rec = STATE["msgs"].get(key)
        if rec:
            return task_resp(rec["taskId"]) if rec["hash"] == h else err(409, "Message id reused.", "IDEMPOTENCY_CONFLICT")
        fl = INFLIGHT.get(key)
        if fl is None:
            INFLIGHT[key] = (h, threading.Event())
    if fl is not None:
        if fl[0] != h:
            return err(409, "Message id reused.", "IDEMPOTENCY_CONFLICT")
        fl[1].wait(50)
        with LOCK:
            rec = STATE["msgs"].get(key)
        return task_resp(rec["taskId"]) if rec else err(503, "Try again.")
    try:
        return continuation(p, msg, key, h) if msg.get("taskId") else new_task(p, msg, key, h)
    finally:
        with LOCK:
            INFLIGHT.pop(key)[1].set()


def new_task(p, msg, key, h):
    parts = msg.get("parts") or []
    if len(parts) != 1 or parts[0].get("mediaType") != MT_BATCH or not isinstance(parts[0].get("data"), dict):
        return err(400, "Invalid batch.")
    data = parts[0]["data"]
    pkgs = data.get("packages")
    if not isinstance(pkgs, list) or not pkgs or not data.get("batchId"):
        return err(400, "Invalid batch.")
    try:
        decisions = decide(pkgs)
    except Exception as ex:
        traceback.print_exc()
        log("DECIDE FAILED:", repr(ex))
        return err(502, "Could not analyse the batch.")
    proposals = []
    for pk, d in zip(pkgs, decisions):
        proposals.append({"packageId": pk["packageId"], "actionId": "act_" + uuid.uuid4().hex, **d})
    tid, cid = "task_" + uuid.uuid4().hex, "ctx_" + uuid.uuid4().hex
    task = {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "art_" + uuid.uuid4().hex, "name": "proposals",
                           "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [msg]}
    with LOCK:
        STATE["tasks"][tid] = {"owner": p, "batchId": data["batchId"],
                               "proposals": {x["packageId"]: x for x in proposals}, "task": task}
        STATE["msgs"][key] = {"hash": h, "taskId": tid}
        save()
    return task_resp(tid)


def continuation(p, msg, key, h):
    with LOCK:
        t = STATE["tasks"].get(msg["taskId"])
        if not t or t["owner"] != p:
            return err(404)
        task = t["task"]
        if task["contextId"] != msg.get("contextId"):
            return err(400, "Context mismatch.")
        if task["status"]["state"] in TERMINAL:
            return err(409, "Task is finished.")
        parts = msg.get("parts") or []
        if len(parts) != 1 or parts[0].get("mediaType") != MT_RES:
            return err(400, "Invalid results.")
        d = parts[0].get("data") or {}
        res = d.get("results")
        if d.get("batchId") != t["batchId"] or not isinstance(res, list) or len(res) != len(t["proposals"]):
            return err(400, "Batch mismatch.")
        seen, execs = set(), []
        for r in res:
            pr = t["proposals"].get(r.get("packageId"))
            if (not pr or r["packageId"] in seen or r.get("actionId") != pr["actionId"]
                    or r.get("action") != pr["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "Result mismatch.")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                if not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]:
                    return err(400, "Missing receipt.")
                execs.append({"packageId": pr["packageId"], "actionId": pr["actionId"], "action": pr["action"],
                              "receiptNonce": r["receiptNonce"], "facts": pr["facts"], "evidenceRefs": pr["evidenceRefs"]})
        task["history"].append(msg)
        task["artifacts"].append({"artifactId": "art_" + uuid.uuid4().hex, "name": "receipts",
                                  "parts": [{"mediaType": MT_REC, "data": {"batchId": t["batchId"], "executions": execs}}]})
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        STATE["msgs"][key] = {"hash": h, "taskId": task["id"]}
        save()
    return task_resp(task["id"])


@app.get(BASE_PATH + "/tasks/<tid>")
def get_task(tid):
    p, e = check(False)
    if e:
        return e
    with LOCK:
        t = STATE["tasks"].get(tid)
        return reply(t["task"]) if t and t["owner"] == p else err(404)


@app.get(BASE_PATH + "/tasks")
def list_tasks():
    p, e = check(False)
    if e:
        return e
    with LOCK:
        return reply({"tasks": [t["task"] for t in STATE["tasks"].values() if t["owner"] == p]})


@app.post(BASE_PATH + "/tasks/<tid>:cancel")
def cancel(tid):
    p, e = check(False)
    if e:
        return e
    with LOCK:
        t = STATE["tasks"].get(tid)
        if not t or t["owner"] != p:
            return err(404)
        task = t["task"]
        if task["status"]["state"] in TERMINAL:
            return err(409, "Task is finished.")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save()
        return reply(task)
