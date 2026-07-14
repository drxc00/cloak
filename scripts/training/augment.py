#!/usr/bin/env python3
"""
Generate synthetic PII-bearing texts in Cloak's target domains.

ai4privacy is mostly prose (emails, narratives), but Cloak's real inputs are
logs, .env blocks, stack traces, git diffs, JSON payloads, Kubernetes events,
CI/CD output, SQL results, Docker logs, Terraform output, and AWS CLI output.
This script creates seeded synthetic rows with exact gold-spans so the
classifier learns to recognise NAME / ADDRESS / USERNAME in machine-shaped
text — and learns hard negatives: company names, hostnames, file paths,
function names that must *not* be redacted.

Every factory has 5-8 template variants so the model doesn't overfit to a
single text pattern.  The weighting targets ~10-15% synthetic / total train.

Usage:
  python scripts/augment.py --out ./augmented/ --rows 10000 --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path

_NAMES_DIR = Path(__file__).resolve().parent / "datasets" / "names"
_USERNAMES_DIR = Path(__file__).resolve().parent / "datasets" / "usernames"

_FALLBACK_FIRST_NAMES = [
    "James",
    "Mary",
    "Wei",
    "Priya",
    "Carlos",
    "Fatima",
    "Liam",
    "Sofia",
    "Amir",
    "Yuki",
    "Olga",
    "Dipesh",
    "Chiara",
    "Bjorn",
    "Saanvi",
]
_FALLBACK_USERNAMES = ["jdoe", "asmith", "devops_ninja", "root_admin", "ci_runner"]


def _load_wordlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )


def _extract_username_cores(
    path: Path, min_len: int = 4, max_len: int = 20
) -> list[str]:
    if not path.exists():
        return []
    pattern = re.compile(rf"[A-Za-z0-9_.]{{{min_len},{max_len}}}")
    cores: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        runs = pattern.findall(line)
        if runs:
            cores.add(max(runs, key=len))
    return sorted(cores)


FIRST_NAMES = _load_wordlist(_NAMES_DIR / "male.txt") + _load_wordlist(
    _NAMES_DIR / "female.txt"
)
if not FIRST_NAMES:
    FIRST_NAMES = _FALLBACK_FIRST_NAMES

LAST_NAMES = FIRST_NAMES  # ~7900 names — classifier learns token-shape, not semantics

_CURATED_USERNAMES = [
    "jdoe",
    "asmith",
    "bwilliams",
    "techlead77",
    "devops_ninja",
    "root_admin",
    "sre_oncall",
    "ml_engineer",
    "data_analyst",
    "ci_runner",
    "deploy_bot",
    "monitoring",
    "apache_user",
    "k8s_admin",
    "ghost_user",
    "backup_svc",
    "mjones",
    "tester01",
    "buildkite",
    "sysop",
    "sa_deploy",
    "db_admin",
    "release_engineer",
    "oncall_primary",
    "secops",
    "infra_bot",
    "pipeline_runner",
    "alerts_svc",
]


def _load_game_usernames(path: Path, sample: int, seed: int) -> list[str]:
    """Load a plain one-per-line gamer-tag dump (Hypixel/Epicube player lists).

    Unlike names/usernames.txt these are already clean ASCII, one token per line —
    no regex extraction needed, just a charset filter + subsample. Millions of lines
    each (2.68M / 751k) so cap the pool: a few thousand is enough extra token-shape
    diversity (digits, underscores, mixed case, decorative padding like leading zeros)
    without one gamer-tag corpus dominating the substitution pool over curated names.
    """
    if not path.exists():
        return []
    shape = re.compile(r"[A-Za-z0-9_]{3,20}")
    candidates = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if shape.fullmatch(ln.strip())
    ]
    if sample and len(candidates) > sample:
        candidates = random.Random(seed).sample(candidates, sample)
    return candidates


USERNAMES = (
    _CURATED_USERNAMES
    + _extract_username_cores(_NAMES_DIR / "usernames.txt")
    + _load_game_usernames(_USERNAMES_DIR / "epicube-players.txt", sample=8000, seed=42)
    + _load_game_usernames(_USERNAMES_DIR / "hypixel-players.txt", sample=8000, seed=42)
)
if not USERNAMES:
    USERNAMES = _FALLBACK_USERNAMES

COMPANY_NAMES = [
    "META",
    "AWS",
    "GCP",
    "Azure",
    "Cloudflare",
    "Datadog",
    "Sentry",
    "Postgres",
    "MySQL",
    "Redis",
    "MongoDB",
    "Elasticsearch",
    "Kafka",
    "Docker",
    "Kubernetes",
    "Terraform",
    "GitHub",
    "GitLab",
    "Jenkins",
    "Prometheus",
    "Grafana",
    "Nginx",
    "Vercel",
    "Netlify",
    "Heroku",
    "Snowflake",
    "BigQuery",
    "Looker",
    "Tableau",
    "Airflow",
    "dbt",
    "Notion",
    "Slack",
    "Linear",
    "Figma",
    "Stripe",
    "Twilio",
    "SendGrid",
    "OpenAI",
    "Anthropic",
    "Cohere",
    "HuggingFace",
    "PyTorch",
    "FastAPI",
    "Okta",
    "Auth0",
    "Supabase",
    "PlanetScale",
    "Vercel",
    "Railway",
    "Fly.io",
    "Render",
    "Bun",
    "Deno",
    "Clerk",
    "Resend",
    "Upstash",
    "Cloudinary",
    "Mapbox",
    "Algolia",
    "Pinecone",
    "Weaviate",
]

TECH_TOKENS = [
    "main",
    "master",
    "develop",
    "feature",
    "bugfix",
    "hotfix",
    "release",
    "init",
    "setup",
    "configure",
    "authenticate",
    "authorize",
    "validate",
    "serialize",
    "deserialize",
    "transform",
    "aggregate",
    "normalize",
    "handle_request",
    "process_batch",
    "enqueue_job",
    "read_config",
    "write_cache",
    "flush_buffer",
    "rotate_logs",
    "health_check",
    "retry_with_backoff",
    "reconcile_state",
    "sync_inventory",
    "refresh_tokens",
    "invalidate_session",
    "purge_artifacts",
    "validate_checksum",
    "resolve_alias",
    "emit_metric",
    "backfill_partitions",
    "rotate_secrets",
    "drain_queue",
    "compact_index",
    "rebalance_shards",
    "throttle_traffic",
]

STREETS = [
    "742 Evergreen Terrace",
    "221B Baker Street",
    "10 Downing Street",
    "1600 Pennsylvania Avenue",
    "1 Infinite Loop",
    "350 Fifth Avenue",
    "4059 Mt Lee Drive",
    "100 Market Street",
    "45 Rockefeller Plaza",
    "200 Larkin Street",
    "1200 12th Avenue South",
    "88 Colin P Kelly Jr St",
    "4 Privet Drive",
    "12 Grimmauld Place",
    "17 Cherry Tree Lane",
    "1313 Mockingbird Lane",
    "112 Ocean Avenue",
    "924 Bel Air Road",
    "890 Fifth Avenue",
    "360 Nueces Street",
]

CITIES_ZIPS = [
    ("Springfield", "62704"),
    ("New York", "10001"),
    ("San Francisco", "94105"),
    ("Seattle", "98101"),
    ("Austin", "73301"),
    ("London", "SW1A 1AA"),
    ("Berlin", "10115"),
    ("Tokyo", "100-0001"),
    ("Sydney", "2000"),
    ("Toronto", "M5V 2T6"),
    ("Paris", "75001"),
    ("Amsterdam", "1012"),
    ("Mumbai", "400001"),
    ("Singapore", "018989"),
    ("Dublin", "D01"),
    ("Stockholm", "111 20"),
    ("Oslo", "0150"),
    ("Copenhagen", "1050"),
    ("Helsinki", "00100"),
    ("Lisbon", "1100"),
    ("Chicago", "60601"),
    ("Denver", "80202"),
    ("Atlanta", "30301"),
    ("Portland", "97201"),
    ("Boston", "02101"),
]

HOSTNAME_LIKE = [
    "api-gateway",
    "auth-service",
    "db-primary",
    "cache-layer",
    "worker-1",
    "lb-public",
    "bastion-host",
    "monitoring-stack",
    "us-east-1",
    "eu-west-2",
    "prod-cluster",
    "staging-vm",
    "ip-10-0-1-42",
    "k8s-node-pool-a",
    "jenkins-agent",
    "redis-master-0",
    "pg-replica-3",
    "es-data-warm",
    "thanos-sidecar",
    "cert-manager",
    "ingress-controller",
    "fluentd-aggregator",
    "argo-rollouts",
    "linkerd-proxy",
    "vault-active",
    "consul-server",
    "traefik-lb",
]

_TLDS = ["com", "io", "dev", "co", "net", "org", "app"]


def _make_span(
    text: str, start: int, end: int, label: str, value: str | None = None
) -> dict:
    return {
        "label": label,
        "start": start,
        "end": end,
        "value": value or text[start:end],
    }


def _find_spans(text: str, needles: list[tuple[str, str]]) -> list[dict]:
    spans: list[dict] = []
    for needle, label in needles:
        for m in re.finditer(re.escape(needle), text):
            spans.append(_make_span(text, m.start(), m.end(), label))
    return spans


def _apply_casing(name: str, style: str) -> str:
    if style == "upper":
        return name.upper()
    if style == "lower":
        return name.lower()
    return name


def _rand_ip(rng: random.Random) -> str:
    return f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"


def _rand_ts(rng: random.Random) -> str:
    return (
        f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        f"T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}Z"
    )


def _stack_trace(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    fn = rng.choice(TECH_TOKENS)
    city, zipcode = rng.choice(CITIES_ZIPS)
    street = rng.choice(STREETS)
    exc = rng.choice(
        [
            "AuthError",
            "ConnectionError",
            "TimeoutError",
            "ValidationError",
            "PermissionDenied",
            "RateLimitExceeded",
        ]
    )

    templates = [
        f"Traceback (most recent call last):\n"
        f'  File "/home/{u}/app/{fn}.py", line 42, in {fn}\n'
        f"    result = client.process(data)\n"
        f'  File "/opt/{c}/sdk/client.py", line 128, in process\n'
        f"    raise {exc}(f'User {f} {l} not authorized')\n"
        f"cloak.{exc}: User {f} {l} not authorized for {h}\n"
        f"  at {street}, {city} {zipcode}",
        f"Panic: runtime error in {fn}\n"
        f"goroutine 42 [running]:\n"
        f"  {c}/sdk.Client.Process(0xdeadbeef)\n"
        f"    /go/pkg/{c}/sdk/client.go:217 +0x1a3\n"
        f"  main.{fn}(0xc0000a4000)\n"
        f"    /home/{u}/app/main.go:89 +0x5f\n"
        f"  created by {f}.{l}@{c}.com",
        f"FATAL: Unhandled rejection in {h}\n"
        f"  at RequestContext.validateAuth (/opt/{c}/auth.js:142:11)\n"
        f"  at async RouteHandler (/opt/{c}/routes.js:33:5)\n"
        f"  user={f} {l}  session=expired  retries=3",
        f"{exc} at {h}.{c}.com\n"
        f"  caused by: java.lang.IllegalStateException: user {f}.{l} not in allowed group\n"
        f"    at com.{c.lower()}.auth.GroupFilter.apply(GroupFilter.java:87)\n"
        f"  requestId=req-{rng.randint(1000,9999)}  principal={u}@{c}.io",
        f"[{h}] ERROR {fn} → {exc}: session expired for {f} {l}\n"
        f"  trace_id={rng.randint(100000,999999):x}\n"
        f"  span=auth.validate  duration_ms={rng.randint(10,500)}\n"
        f"  metadata={{user: {u}, org: {c}, ip: {_rand_ip(rng)}}}",
    ]
    text = rng.choice(templates)

    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
            (street, "ADDRESS"),
            (city, "ADDRESS"),
            (zipcode, "ADDRESS"),
        ],
    )


def _env_block(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    city, _ = rng.choice(CITIES_ZIPS)

    templates = [
        f"# Deployed by: {f} {l}\n"
        f"export APP_ENV=production\n"
        f"export REGION=us-east-1\n"
        f"export COMPANY={c}\n"
        f"export DB_USER={u}\n"
        f"export DB_HOST=db-primary.{c}.internal\n"
        f"export REDIS_HOST=cache-layer.prod.{c}.io\n"
        f'export APP_NAME="{c} Dashboard"\n'
        f"export DEPLOY_USER={f}_{l}\n"
        f"export OFFICE_LOCATION={city}",
        f"## OWNER: {f} {l}\n"
        f"DATABASE_URL=postgres://{u}:redacted@{c}-db.internal:5432/app\n"
        f"REDIS_URL=redis://{c}-cache.prod.{c}.io:6379\n"
        f"LOG_LEVEL=debug\n"
        f"ADMIN_EMAIL={f}.{l}@{c}.io\n"
        f"DEPLOY_REGION={city}",
        f"# Managed by {u}\n"
        f"TF_VAR_owner={f} {l}\n"
        f"TF_VAR_region={city}\n"
        f"TF_VAR_company={c}\n"
        f"TF_VAR_enable_monitoring=true",
        f"GIT_AUTHOR_NAME={f} {l}\n"
        f"GIT_AUTHOR_EMAIL={f}.{l}@{c}.dev\n"
        f"NPM_REGISTRY=https://registry.{c}.io\n"
        f"DOCKER_REGISTRY=registry.{c}.com\n"
        f"KUBECONFIG=/home/{u}/.kube/{c}-prod.yaml",
        f"# secrets for {c} production\n"
        f"export BASTION_USER={u}\n"
        f"export BASTION_HOST=bastion.{c}.io\n"
        f"export ONCALL_NAME={f} {l}\n"
        f"export ONCALL_SLACK=@{u}",
    ]
    text = rng.choice(templates)

    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "_" + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
            (city, "ADDRESS"),
        ],
    )

def _nginx_log(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    ip = _rand_ip(rng)
    ts = _rand_ts(rng)
    method = rng.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])
    path = rng.choice(
        [
            "/api/v1/users",
            "/api/v1/auth",
            "/health",
            f"/api/v1/{c}/config",
            f"/admin/users/{u}",
            "/graphql",
            "/api/v2/events",
        ]
    )
    status = rng.choice([200, 200, 200, 201, 301, 400, 401, 403, 500])
    ua = rng.choice(
        [
            f"Mozilla/5.0 ({f} {l}; {c}Bot/1.0)",
            f"curl/7.88.1 ({f}.{l})",
            f"python-requests/2.31 ({u})",
            f"Go-http-client/2.0 ({c}-sdk)",
        ]
    )

    templates = [
        f'{ip} - {u} [{ts}] "{method} {path} HTTP/1.1" {status} 1234 '
        f'"https://{h}.{c}.io/dashboard" "{ua}"',
        f'{ip} - - [{ts}] "{method} {path} HTTP/2" {status} 567 '
        f'"https://{h}.{c}.com/admin?user={f}_{l}" '
        f'"Mozilla/5.0 ({c} Internal)"',
        f'{ip} {f}.{l} [{ts}] "{method} {path} HTTP/1.1" {status} 892 ' f'"{ua}" "-"',
        f'{ip} - {u} [{ts}] "GET /api/v2/audit?principal={f}%20{l} HTTP/1.1" 200 442 '
        f'"https://{c}.io/admin/audit" "{ua}"',
    ]
    text = rng.choice(templates)

    return text, _find_spans(
        text,
        [
            (u, "USERNAME"),
            (f + " " + l, "NAME"),
            (f + "_" + l, "NAME"),
            (f + "." + l, "NAME"),
        ],
    )

def _git_diff(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    date = f"2025-0{rng.randint(1,9)}-{rng.randint(1,28):02d}"

    templates = [
        f"commit a1b2c3d4e5f6\n"
        f"Author: {f} {l} <{u}@{c}.com>\n"
        f"Date:   Mon Jul 7 10:30:00 2025 +0000\n"
        f"\n"
        f"    fix: update auth config for {h}\n"
        f"\n"
        f"diff --git a/config/auth.yaml b/config/auth.yaml\n"
        f"@@ -12,6 +12,6 @@\n"
        f"   timeout: 30\n"
        f"   providers:\n"
        f"-    - name: {c}-sso\n"
        f"-      issuer: https://{c}.okta.com\n"
        f"+    - name: {c}-sso-v2\n"
        f"+      issuer: https://{c}.auth0.com\n"
        f"   users:\n"
        f"-    - {u}\n"
        f"+    - {f}.{l}\n",
        f"commit f7e8d9c0\n"
        f"Author: {u} <{u}@{c}.dev>\n"
        f"Date:   {date}\n"
        f"\n"
        f"    feat: add {h} health check endpoint\n"
        f"\n"
        f"diff --git a/src/routes/health.ts b/src/routes/health.ts\n"
        f"+// Added by {f} {l} for {c} infra team\n"
        f"+router.get('/health', async (ctx) => {{\n"
        f"+  ctx.body = {{ status: 'ok', hostname: '{h}' }};\n"
        f"+}});\n",
        f"commit deadbeef\n"
        f"Merge: a1b2c3d 9f8e7d6\n"
        f"Author: {f} {l} <{f}.{l}@{c}.com>\n"
        f"Date:   {date}\n"
        f"\n"
        f"    Merge pull request #4291 from {u}/{h}-fix\n"
        f"\n"
        f"    Reviewed-by: {f} {l}\n"
        f"    Tested-by: {u}",
    ]
    text = rng.choice(templates)

    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
        ],
    )

def _hard_negatives(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    c = rng.choice(COMPANY_NAMES)
    c2 = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    fn = rng.choice(TECH_TOKENS)

    texts = [
        f"My name is {f} {l}. I work at {c}.",
        f"Debugging the {c} SDK with {f} {l}.",
        f"Deploy {h} to {c} — reviewed by {f} {l}.",
        f"Fixing {c2} integration for {f} {l} at {c}.",
        f"Stack trace from {h}: {fn} failed for user {f} {l}.",
        f"Connecting {h} to {c} {c2} — oncall: {f} {l}.",
        f"Onboarding {f} {l} to {c}: granted {h} access.",
        f"Runbook: if {h} alerts, page {f} {l} (team {c}).",
        f"PR #4291 by {f} {l} — adds {c2} provider to {h}.",
        f"Retro: {h} outage — root cause found by {f} {l} ({c}).",
        f"Standup: {f} {l} working on {c} → {c2} migration ({h}).",
        f"PagerDuty: {h} alert assigned to {f} {l} (secondary: {c}-oncall).",
    ]
    text = rng.choice(texts)
    needle = f + " " + l
    spans = [
        _make_span(text, m.start(), m.end(), "NAME")
        for m in re.finditer(re.escape(needle), text)
    ]
    return text, spans

def _stack_trace_upper(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES).upper()
    l = rng.choice(LAST_NAMES).upper()
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    fn = rng.choice(TECH_TOKENS)

    text = (
        f"ERROR: AuthFailure in {fn}\n"
        f"  module=sdk.client  user={f} {l}\n"
        f"  host={h}.{c}.io  region=us-east-1\n"
        f"  detail=User {f} {l} not authorized\n"
        f"  account={u}"
    )
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _log_output_upper(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES).upper()
    l = rng.choice(LAST_NAMES).upper()
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"[INFO] ACCESS_GRANTED USER={f}.{l} ROLE=admin SERVICE={h}",
        f'audit: login ok user="{f} {l}" src=10.0.1.42 dest={h}.{c}.com',
        f"ACCOUNT LOCKED: {f} {l} ({u}) — too many attempts from {h}",
        f"SECURITY: password reset for {f}_{l} requested by {u} on {h}",
        f"DEACTIVATED USER ACCOUNT {f} {l} ({u}) — reason: offboarding",
        f"WARN  UserSync: skipping {f} {l} — not in {c} IdP",
        f"CRIT  {h}: token expired for {f}_{l}, re-auth required",
        f"NOTICE: access granted to {f} {l} for {c}/{h} by {u}",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (f + "_" + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _env_upper_names(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES).upper()
    l = rng.choice(LAST_NAMES).upper()
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    city, _ = rng.choice(CITIES_ZIPS)

    text = (
        f"# DEPLOYMENT METADATA\n"
        f"DEPLOYED_BY={f} {l}\n"
        f"ONCALL_ENGINEER={f}_{l}\n"
        f"APPROVER={f}.{l}@{c}.com\n"
        f"RELEASE_USER={u}\n"
        f"OFFICE={city}"
    )
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "_" + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
            (city, "ADDRESS"),
        ],
    )


def _lowercase_names(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES).lower()
    l = rng.choice(LAST_NAMES).lower()
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"assigning ticket eng-4421 to {f} {l} ({u})",
        f"cc: {f}.{l}@{c}.{rng.choice(_TLDS)}",
        f"oncall handoff: {f} {l} taking over {h} from {u}",
        f"reviewer: {f}_{l} approved pr #2819 for {c}/{h}",
        f"escalating {h} alert to {f} {l} — acknowledged by {u}",
        f"primary contact for {c} infra: {f} {l} / {u}",
        f"added {f} {l} as maintainer of {c}/{h} (requested by {u})",
        f"dm from {u}: hey {f} {l}, can you check the {h} deploy?",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (f + "_" + l, "NAME"),
            (u, "USERNAME"),
        ],
    )

def _hard_negatives_mixed_case(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    casing = rng.choice(["upper", "lower"])
    fc = _apply_casing(f, casing)
    lc = _apply_casing(l, casing)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"ALERT: user {fc} {lc} triggered rate-limit on {h} ({c} infra)",
        f"Deploying {c} SDK v2.4.1 — requester: {fc} {lc}",
        f"CRITICAL: {h} down — oncall is {fc} {lc} (team {c})",
        f"Access review: {fc} {lc} has admin on {c}/{h}",
    ]
    text = rng.choice(templates)
    needle = fc + " " + lc
    spans = [
        _make_span(text, m.start(), m.end(), "NAME")
        for m in re.finditer(re.escape(needle), text)
    ]
    return text, spans

def _json_log(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    city, _ = rng.choice(CITIES_ZIPS)
    ip = _rand_ip(rng)

    templates = [
        f'{{"ts":"2025-07-14T08:12:33Z","lvl":"info","msg":"login ok",'
        f'"user":"{u}","name":"{f} {l}","ip":"{ip}",'
        f'"svc":"{h}","org":"{c}"}}',
        f'{{"timestamp":"2025-07-14T08:12:33.456Z","level":"warn",'
        f'"message":"rate limit exceeded","actor":"{f} {l}",'
        f'"actor_id":"{u}","resource":"{h}","tenant":"{c}","location":"{city}"}}',
        f'{{"@timestamp":"2025-07-14","severity":"ERROR",'
        f'"body":"auth failure for {f}_{l} on {h}",'
        f'"context":{{"user":"{u}","org":"{c}"}}}}',
        f'{{"log":"access","principal":"{f}.{l}@{c}.io",'
        f'"action":"delete","resource":"{h}","result":"denied",'
        f'"delegated_by":"{u}","src_ip":"{ip}"}}',
        f'{{"event":"audit.log","actor":{{"name":"{f} {l}","id":"{u}"}},'
        f'"target":{{"type":"host","id":"{h}"}},"org":"{c}","outcome":"success"}}',
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "_" + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
            (city, "ADDRESS"),
        ],
    )

def _kubernetes_event(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    ns = rng.choice(["prod", "staging", "dev", f"{c}-infra"])

    templates = [
        f"LAST SEEN   TYPE      REASON              OBJECT                  MESSAGE\n"
        f"12s         Normal    Scheduled           pod/{h}-7d4f     Successfully assigned {ns}/{h}-7d4f to ip-10-0-1-42\n"
        f'8s          Warning   FailedMount         pod/{h}-7d4f     MountVolume.SetUp failed for volume "config" (requested by {u})\n'
        f"3s          Normal    Created             pod/{h}-7d4f     Created container main (image: {c}/app:v2.4.1, requester: {f} {l})",
        f"Events:\n"
        f"  Type     Reason     Age   From               Message\n"
        f'  Normal   Pulling    2m    kubelet            Pulling image "{c}/app:latest"\n'
        f"  Normal   Created    1m    kubelet            Created container app\n"
        f"  Normal   Started    1m    kubelet            Started container app\n"
        f"  Warning  Unhealthy  30s   kubelet            Liveness probe failed: "
        f'Get "http://{h}:8080/health": dial tcp 10.0.1.42:8080: connect refused '
        f"(operator: {f} {l}, namespace: {ns})",
        f"Name:         {h}-7d4f-abcde\n"
        f"Namespace:    {ns}\n"
        f"Priority:     0\n"
        f"Node:         ip-10-0-1-42/{_rand_ip(rng)}\n"
        f"Labels:       app={h}, owner={u}, team={c}-infra\n"
        f"Annotations:  deployed-by={f}_{l}\n"
        f"Status:       Running\n"
        f"IP:           {_rand_ip(rng)}",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "_" + l, "NAME"),
            (u, "USERNAME"),
        ],
    )

def _ci_cd_output(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    sha = f"{rng.randint(0,15):x}{rng.randint(0,15):x}"

    templates = [
        f"Run {c}/deploy (attempt #2)\n"
        f"  triggered by: {u} ({f} {l})\n"
        f"  commit:       {sha}\n"
        f"  branch:       feature/{h}-migration\n"
        f"  ---\n"
        f"  ✓  Build image ({c}/app:{sha[:7]})\n"
        f"  ✓  Push to registry\n"
        f"  ✓  Deploy to {h}\n"
        f"  ✓  Smoke test passed\n"
        f"  ✓  Notify {f}.{l}@{c}.io",
        f"--- deploy ({h}) ---\n"
        f"actor:      {u}\n"
        f"approver:   {f} {l}\n"
        f"env:        production\n"
        f"sha:        {sha}\n"
        f"status:     success\n"
        f"duration:   2m34s\n"
        f"rollback:   disabled\n"
        f"notify:     #{c}-eng, {u}",
        f"Phase 1/5: BUILD    [{h}] ✓  (1m12s)\n"
        f"Phase 2/5: TEST     [{h}] ✓  (2m08s)\n"
        f"Phase 3/5: STAGING  [{h}] ✓  (0m45s) — approved by {u}\n"
        f"Phase 4/5: CANARY   [{h}] ✓  (3m22s)\n"
        f"Phase 5/5: PROD     [{h}] ✓  (1m51s) — deployed by {f} {l}\n"
        f"Summary: 5/5 passed — {c}/{h} at {sha[:7]}",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
        ],
    )

def _sql_result(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    street = rng.choice(STREETS)
    city, zipcode = rng.choice(CITIES_ZIPS)

    templates = [
        f" id | name              | email                    | address\n"
        f"----+-------------------+--------------------------+---------------------------------\n"
        f" 42 | {f} {l}           | {f}.{l}@{c}.com          | {street}, {city} {zipcode}\n"
        f" 43 | {u:17s} | {u}@{c}.io              | \n"
        f"(2 rows)",
        f"-[ RECORD 1 ]------------\n"
        f"id       | 427\n"
        f"username | {u}\n"
        f"name     | {f} {l}\n"
        f"email    | {f}.{l}@{c}.dev\n"
        f"location | {city}\n"
        f"host     | {h}.{c}.com\n"
        f"active   | t",
        f"UPDATE users SET email = '{f}.{l}@{c}.io', "
        f"location = '{city}' WHERE username = '{u}';\n"
        f"UPDATE 1\n"
        f"Time: 0.423 ms",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
            (street, "ADDRESS"),
            (city, "ADDRESS"),
            (zipcode, "ADDRESS"),
        ],
    )

def _slack_webhook(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"<@{u}> {h} is down — paging {f} {l}\n"
        f"> {c} production alert | severity: critical\n"
        f"> runbook: https://wiki.{c}.io/runbooks/{h}",
        f"Incident #482 — {h} latency spike\n"
        f"Commander: {f} {l}\n"
        f"Comms: {u}\n"
        f"Status: investigating — {c} SDK timeout at 2000ms\n"
        f"Timeline:\n"
        f"  08:12  {u} paged by {h} alert\n"
        f"  08:14  {f} {l} joined incident channel",
        f"FYI: {f} {l} is OOO until Friday — {h} oncall coverage by {u}\n"
        f"Contact {c}-infra if anything urgent before then.",
        f"@here can someone approve deploy #{rng.randint(1000,9999)} for {h}?\n"
        f"  requester: {u}\n"
        f"  reviewer:  {f} {l}\n"
        f"  diff:      https://github.com/{c}/{h}/pull/{rng.randint(100,999)}",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _code_comment(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    fn = rng.choice(TECH_TOKENS)

    templates = [
        f"// TODO({u}): refactor this after {h} migration\n"
        f"// Author: {f} {l} <{f}.{l}@{c}.com>\n"
        f"// See: ENG-4291\n"
        f"func {fn}(ctx context.Context, req *Request) (*Response, error) {{\n"
        f"    // HACK: {c} SDK doesn't support streaming yet\n"
        f"    return client.Do(ctx, req)\n"
        f"}}",
        f"# Owner: {f} {l}\n"
        f"# Module: {c} integration for {h}\n"
        f"# NOTE({u}): temporary workaround — remove after {c} v3.0\n"
        f"from {c.lower()}_sdk import Client\n"
        f"\n"
        f"client = Client(host='{h}.{c}.io')",
        f"/**\n"
        f" * {c} health check handler for {h}\n"
        f" * @author {f} {l}\n"
        f" * @see ENG-{rng.randint(1000,9999)}\n"
        f" */\n"
        f"const {fn} = async (req, res) => {{\n"
        f"  // auth delegated to {c} IdP — see /docs/auth.md\n"
        f"  res.json({{ status: 'ok', node: '{h}' }});\n"
        f"}};",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _code_block(rng: random.Random) -> tuple[str, list[dict]]:
    """
    Realistic multi-line source files.  The model must find names in
    docstrings/copyright headers while leaving function names, class names,
    and technical identifiers alone.
    """
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    fn = rng.choice(TECH_TOKENS)
    city, _ = rng.choice(CITIES_ZIPS)

    templates = [
        # Python module
        f'"""\n'
        f"{c} SDK — client library for {h}\n"
        f"\n"
        f"Author: {f} {l} <{f}.{l}@{c}.io>\n"
        f"Copyright (c) 2025 {c}. All rights reserved.\n"
        f'"""\n'
        f"\n"
        f"import logging\n"
        f"from typing import Optional\n"
        f"\n"
        f"logger = logging.getLogger(__name__)\n"
        f"\n"
        f"\n"
        f"class {c}Client:\n"
        f'    """Main entry point for {c} API operations."""\n'
        f"\n"
        f'    def __init__(self, api_key: str, region: str = "us-east-1"):\n'
        f"        self.api_key = api_key\n"
        f"        self.region = region\n"
        f"        self._session = None\n"
        f"\n"
        f"    def {fn}(self, resource_id: str) -> dict:\n"
        f'        """\n'
        f"        Fetch a resource by ID.\n"
        f"\n"
        f"        Args:\n"
        f"            resource_id: The {c} resource identifier.\n"
        f"\n"
        f"        Returns:\n"
        f"            dict with keys: id, status, owner, created_at\n"
        f"\n"
        f"        Raises:\n"
        f"            {c}Error: if the resource is not found or access denied.\n"
        f'        """\n'
        f'        path = f"/v1/resources/{{resource_id}}"\n'
        f"        response = self._get(path)\n"
        f"        if response.status_code == 404:\n"
        f'            raise {c}Error(f"Resource {{resource_id}} not found")\n'
        f"        return response.json()\n"
        f"\n"
        f"    def shutdown(self) -> None:\n"
        f'        """Gracefully close connections and flush metrics."""\n'
        f'        logger.info("Shutting down {c} client for %s", {u!r})\n'
        f"        if self._session:\n"
        f"            self._session.close()\n",
        # Go file
        f"// Package auth provides authentication primitives for {h}.\n"
        f"//\n"
        f"// Author: {f} {l} <{u}@{c}.io>\n"
        f"// Copyright 2025 {c}, Inc.\n"
        f"// Licensed under the MIT License.\n"
        f"package auth\n"
        f"\n"
        f"import (\n"
        f'\t"context"\n'
        f'\t"fmt"\n'
        f'\t"time"\n'
        f"\n"
        f'\t"{c}.io/sdk/internal/cache"\n'
        f")\n"
        f"\n"
        f"// Authenticator validates credentials and issues session tokens.\n"
        f"type Authenticator struct {{}}\n"
        f"\tclient  *http.Client\n"
        f"\tcache   *cache.LRU\n"
        f"\tmetrics *Metrics\n"
        f"}}\n"
        f"\n"
        f"// New returns an Authenticator backed by {c} IAM.\n"
        f"// The instance auto-registers with {h} on creation.\n"
        f"func New(endpoint string) *Authenticator {{}}\n"
        f"\treturn &Authenticator{{}}\n"
        f"\t\tclient:  newHTTPClient(endpoint),\n"
        f"\t\tcache:   cache.NewLRU(1024),\n"
        f'\t\tmetrics: globalMetrics.withTag("component", "{h}"),\n'
        f"\t}}\n"
        f"}}\n"
        f"\n"
        f"// {fn} validates a credential and returns an expiry time.\n"
        f"func (a *Authenticator) {fn}(ctx context.Context, token string) (time.Time, error) {{\n"
        f'\tif token == "" {{\n'
        f'\t\treturn time.Time{{}}, fmt.Errorf("empty token")\n'
        f"\t}}\n"
        f"\treturn a.cache.GetOrFill(ctx, token, a.doValidate)\n"
        f"}}\n",
        # TypeScript config service
        f"/**\n"
        f" * {c} Config Service\n"
        f" *\n"
        f" * Manages runtime configuration for {h} deployments.\n"
        f" * @module config\n"
        f" * @author {f} {l}\n"
        f" * @license {c}-proprietary\n"
        f" */\n"
        f"\n"
        f'import {{ Logger }} from "@pino/logger";\n'
        f'import {{ RedisClient, PostgresPool }} from "@{c}/infra";\n'
        f"\n"
        f'const logger = new Logger({{ name: "{h}-config" }});\n'
        f"\n"
        f"interface AppConfig {{\n"
        f'  env: "production" | "staging" | "development";\n'
        f"  region: string;\n"
        f"  features: Record<string, boolean>;\n"
        f"  owner: string;\n"
        f"}}\n"
        f"\n"
        f"export class ConfigService {{\n"
        f"  private cache: RedisClient;\n"
        f"  private db: PostgresPool;\n"
        f"\n"
        f"  constructor(\n"
        f"    private readonly host: string,\n"
        f"    private readonly version: string,\n"
        f"  ) {{\n"
        f"    this.cache = new RedisClient({{ host: `${{host}}-cache`, port: 6379 }});\n"
        f'    this.db = new PostgresPool({{ host: `${{host}}-db`, database: "{c}" }});\n'
        f"  }}\n"
        f"\n"
        f"  async {fn}(key: string): Promise<AppConfig | null> {{\n"
        f"    const cached = await this.cache.get(key);\n"
        f"    if (cached) return JSON.parse(cached) as AppConfig;\n"
        f"\n"
        f"    const result = await this.db.query(\n"
        f'      "SELECT config FROM {c}.app_configs WHERE key = $1",\n'
        f"      [key]\n"
        f"    );\n"
        f"    if (!result.rows.length) return null;\n"
        f"\n"
        f"    const config = result.rows[0].config as AppConfig;\n"
        f'    await this.cache.set(key, JSON.stringify(config), "EX", 300);\n'
        f"    return config;\n"
        f"  }}\n"
        f"}}\n",
        # Shell script with deploy metadata
        f"#!/usr/bin/env bash\n"
        f"#\n"
        f"# deploy-{h}.sh — Blue/green deploy for {c} {h}\n"
        f"# Author: {f} {l}\n"
        f"# Maintainer: {u}@{c}.io\n"
        f"#\n"
        f"set -euo pipefail\n"
        f"\n"
        f'CLUSTER="${{1:-{h}}}"\n'
        f'NAMESPACE="{c}-prod"\n'
        f'IMAGE_TAG="${{IMAGE_TAG:-latest}}"\n'
        f"\n"
        f'echo "=== Deploying $IMAGE_TAG to $CLUSTER ($NAMESPACE) ==="\n'
        f"\n"
        f"# Scale up green deployment\n"
        f'kubectl scale deployment "$CLUSTER-green" \\\n'
        f'  --namespace="$NAMESPACE" \\\n'
        f"  --replicas=3\n"
        f"\n"
        f'echo "Waiting for green pods..."\n'
        f"kubectl wait --for=condition=ready pod \\\n"
        f'  -l "app=$CLUSTER,color=green" \\\n'
        f'  --namespace="$NAMESPACE" \\\n'
        f"  --timeout=300s\n"
        f"\n"
        f"# Health check\n"
        f'GREEN_URL="https://$CLUSTER-green.{c}.io/health"\n'
        f'HTTP_CODE=$(curl -s -o /dev/null -w "%{{http_code}}" "$GREEN_URL")\n'
        f'if [[ "$HTTP_CODE" != "200" ]]; then\n'
        f'  echo "FATAL: green health check returned $HTTP_CODE"\n'
        f"  exit 1\n"
        f"fi\n"
        f"\n"
        f'echo "✓ Green healthy — deploying by {u}"\n'
        f'echo "✓ Notify {f}.{l}@{c}.io on completion"\n',
        # Dockerfile
        f"# syntax=docker/dockerfile:1\n"
        f"# {c} {h} runtime image\n"
        f"# Maintainer: {f} {l} <{f}.{l}@{c}.io>\n"
        f"\n"
        f"FROM node:22-alpine AS builder\n"
        f"WORKDIR /app\n"
        f"COPY package*.json ./\n"
        f"RUN npm ci --omit=dev\n"
        f"COPY . .\n"
        f"RUN npm run build\n"
        f"\n"
        f"FROM node:22-alpine\n"
        f'LABEL org.opencontainers.image.authors="{f} {l}"\n'
        f'LABEL org.opencontainers.image.vendor="{c}"\n'
        f'LABEL com.{c.lower()}.service="{h}"\n'
        f"\n"
        f"RUN addgroup -S app && adduser -S {u} -G app\n"
        f"WORKDIR /app\n"
        f"COPY --from=builder /app/dist ./dist\n"
        f"COPY --from=builder /app/node_modules ./node_modules\n"
        f"\n"
        f"USER {u}\n"
        f"EXPOSE 8080\n"
        f"HEALTHCHECK --interval=30s CMD wget -qO- http://localhost:8080/health || exit 1\n"
        f'CMD ["node", "dist/{h}/server.js"]\n',
        # YAML config
        f"# {c} {h} service configuration\n"
        f"# Owner: {f} {l}\n"
        f"# Last updated: 2025-07-14 by {u}\n"
        f"\n"
        f"service:\n"
        f"  name: {h}\n"
        f'  version: "2.4.1"\n'
        f"  namespace: {c}\n"
        f"\n"
        f"auth:\n"
        f"  provider: {c}-iam\n"
        f"  issuer: https://{c}.okta.com/oauth2/default\n"
        f"  client_id: ${{{{{c}_CLIENT_ID}}}}\n"
        f"\n"
        f"database:\n"
        f"  host: db-primary.{c}.internal\n"
        f"  name: {h}\n"
        f"  pool_size: 20\n"
        f"  ssl: true\n"
        f"\n"
        f"monitoring:\n"
        f"  datadog:\n"
        f"    enabled: true\n"
        f"    tags:\n"
        f'      - "service:{h}"\n'
        f'      - "team:{c}-infra"\n'
        f'      - "owner:{u}"\n'
        f"\n"
        f"owners:\n"
        f"  primary: {f} {l}\n"
        f"  secondary: {u}\n"
        f"  team: {c}-eng\n",
        # SQL migration
        f"-- Migration: add {c} user audit table\n"
        f"-- Author: {f} {l}\n"
        f"-- Date: 2025-07-14\n"
        f"-- Jira: ENG-{rng.randint(1000,9999)}\n"
        f"\n"
        f"BEGIN;\n"
        f"\n"
        f"CREATE TABLE IF NOT EXISTS {c}.user_audit (\n"
        f"    id          BIGSERIAL PRIMARY KEY,\n"
        f"    user_id     VARCHAR(64)  NOT NULL,\n"
        f"    action      VARCHAR(32)  NOT NULL,\n"
        f"    resource    VARCHAR(256),\n"
        f"    old_value   JSONB,\n"
        f"    new_value   JSONB,\n"
        f"    ip_address  INET,\n"
        f"    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()\n"
        f");\n"
        f"\n"
        f"COMMENT ON TABLE {c}.user_audit IS 'Audit log for {c} {h} user actions.';\n"
        f"\n"
        f"CREATE INDEX idx_user_audit_user_id\n"
        f"    ON {c}.user_audit (user_id, created_at DESC);\n"
        f"\n"
        f"-- Seed initial data\n"
        f"INSERT INTO {c}.user_audit (user_id, action, resource, ip_address) "
        f"VALUES ('{u}', 'CREATE', '{h}', '10.0.1.42');\n"
        f"\n"
        f"COMMIT;\n",
    ]
    text = rng.choice(templates)

    # Names appear in comments/docstrings across all templates, plus in YAML
    # owners section and Dockerfile labels.  Usernames appear in shell vars,
    # SQL values, Docker USER, and YAML tags.
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (f + "." + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _docker_log(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"[{h}] 2025-07-14T08:12:33Z INFO  Starting {c} app v2.4.1\n"
        f"[{h}] 2025-07-14T08:12:34Z INFO  Loading config from /etc/{c}/config.yaml\n"
        f"[{h}] 2025-07-14T08:12:35Z INFO  Connecting to {c} PostgreSQL at db-primary.{c}.internal:5432\n"
        f"[{h}] 2025-07-14T08:12:36Z INFO  Authenticated as {u} ({f} {l})\n"
        f"[{h}] 2025-07-14T08:12:37Z INFO  Listening on :8080",
        f"{h}        | [WARN] rate limiting enabled — 100 req/s per user\n"
        f"{h}        | [INFO] admin session created for {u}\n"
        f"{h}        | [INFO] operator oncall: {f} {l} ({u})\n"
        f"{h}        | [INFO] metrics exported to {c} Datadog",
        f"CONTAINER ID   IMAGE                  CREATED        STATUS        NAMES\n"
        f"a1b2c3d4e5f6   {c}/app:v2.4.1          2 mins ago     Up 2 mins     {h}\n"
        f"f7e8d9c0a1b2   {c}/worker:v1.9.0       5 hours ago    Up 5 hours    {h}-worker\n"
        f"            deployed by: {f} {l} ({u})",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _terraform_output(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"Terraform v1.7.4\n"
        f"on tier1/{c}/{h}/main.tf\n"
        f"\n"
        f"Initializing provider plugins...\n"
        f'- Finding hashicorp/aws versions matching "~> 5.0"...\n'
        f"- Installing hashicorp/aws v5.42.0...\n"
        f"\n"
        f"Plan: 3 to add, 1 to change, 0 to destroy.\n"
        f"\n"
        f"Changes:\n"
        f"  + module.{h}.aws_instance.app\n"
        f"      tags = {{\n"
        f'        Owner   = "{f} {l}"\n'
        f'        Managed = "{u}"\n'
        f'        Team    = "{c}-infra"\n'
        f"      }}",
        f"Apply complete! Resources: 3 added, 1 changed, 0 destroyed.\n"
        f"Outputs:\n"
        f'  {h}_endpoint = "https://{h}.{c}.io"\n'
        f'  operator      = "{f} {l}"\n'
        f'  deployed_by   = "{u}"',
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _aws_cli_output(rng: random.Random) -> tuple[str, list[dict]]:
    f = rng.choice(FIRST_NAMES)
    l = rng.choice(LAST_NAMES)
    u = rng.choice(USERNAMES)
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)

    templates = [
        f"{{\n"
        f'    "User": {{\n'
        f'        "Path": "/",\n'
        f'        "UserName": "{u}",\n'
        f'        "UserId": "AIDA{chr(65+rng.randint(0,25))}{rng.randint(1000,9999)}",\n'
        f'        "Arn": "arn:aws:iam::{rng.randint(100000000000,999999999999):12d}:user/{u}",\n'
        f'        "Tags": [\n'
        f'            {{"Key": "Name", "Value": "{f} {l}"}},\n'
        f'            {{"Key": "Team", "Value": "{c}-infra"}},\n'
        f'            {{"Key": "Host", "Value": "{h}"}}\n'
        f"        ]\n"
        f"    }}\n"
        f"}}",
        f"{{\n"
        f'    "Reservations": [\n'
        f"        {{\n"
        f'            "Instances": [\n'
        f"                {{\n"
        f'                    "InstanceId": "i-{rng.randint(0,15):x}{rng.randint(0,15):x}",\n'
        f'                    "Tags": [\n'
        f'                        {{"Key": "Name", "Value": "{h}"}},\n'
        f'                        {{"Key": "Owner", "Value": "{f} {l}"}},\n'
        f'                        {{"Key": "ManagedBy", "Value": "{u}"}}\n'
        f"                    ]\n"
        f"                }}\n"
        f"            ]\n"
        f"        }}\n"
        f"    ]\n"
        f"}}",
    ]
    text = rng.choice(templates)
    return text, _find_spans(
        text,
        [
            (f + " " + l, "NAME"),
            (u, "USERNAME"),
        ],
    )


def _pure_o_hard_negatives(rng: random.Random) -> tuple[str, list[dict]]:
    c = rng.choice(COMPANY_NAMES)
    h = rng.choice(HOSTNAME_LIKE)
    fn = rng.choice(TECH_TOKENS)
    ip = _rand_ip(rng)
    ts = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    sha = f"{rng.randint(0,15):x}{rng.randint(0,15):x}"

    templates = [
        f"[{ts}T08:00:00Z] INFO  {h} {fn} → status=ok latency=12ms region=us-east-1",
        f"Deploying {c} SDK v2.4.1 to {h}.{c}.io — release notes: https://{c}.dev/changelog",
        f"Connecting {h} to {c} PostgreSQL — port 5432, pool size 20",
        f"Pulling {c}/{fn}:latest from registry.{c}.io — sha256:{sha}",
        f"Terraform apply: {c}-infra/{h} — 3 resources changed, 0 destroyed",
        f"Running {fn} on {ip} — allocated 512MB, pid 8842",
        f"Reloading {c} config from /etc/{c}/config.yaml — {h} section updated",
        f"Health check: {h}.{c}.com → 200 OK (0.042s)",
        f"Backup complete: {h}/{c}-db → s3://{c}-backups/{ts}/dump.sql.gz (234MB)",
        f"Renewing TLS cert for {h}.{c}.io — issued by LetsEncrypt, valid until 2025-10-01",
        f"Cache warm: {h} → {c} Redis (6379) loaded 12,847 keys in 3.2s",
        f"Rolling restart: {h} pods 1-3 → draining connections (GracefulShutdown={fn})",
        f"Pushing {c}/{h}:{sha[:7]} to registry.{c}.io — 142MB in 3 layers",
        f"Running migration {sha[:8]}_add_index on {c}-db — applied in 234ms",
        f"[{h}] Scaling from 3 to 5 replicas (CPU at 78% for 5m)",
    ]
    text = rng.choice(templates)
    return text, []  # no spans — pure O


# ---------------------------------------------------------------------------
# Weighted factory registry (~15% of 70k ai4privacy train = ~10k rows)
# ---------------------------------------------------------------------------

FACTORIES = [
    # Original title-case  (~30%)
    (_stack_trace, 10),
    (_env_block, 8),
    (_nginx_log, 10),
    (_git_diff, 8),
    (_hard_negatives, 16),
    # Case variants  (~15%)
    (_stack_trace_upper, 4),
    (_log_output_upper, 4),
    (_env_upper_names, 3),
    (_lowercase_names, 5),
    (_hard_negatives_mixed_case, 4),
    # New domain categories  (~25%)
    (_json_log, 7),
    (_kubernetes_event, 5),
    (_ci_cd_output, 5),
    (_sql_result, 4),
    (_slack_webhook, 4),
    (_code_comment, 4),
    (_docker_log, 4),
    (_terraform_output, 4),
    (_aws_cli_output, 4),
    # Full code blocks  (~12%)
    (_code_block, 10),
    # Pure-O hard negatives  (~14%)
    (_pure_o_hard_negatives, 10),
]
# Total weight: 133 → 10,000 rows = ~75 rows per weight point


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate domain-shaped synthetic PII rows"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for .jsonl files"
    )
    parser.add_argument(
        "--rows", type=int, default=10000, help="Total synthetic rows to generate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    factory_names, weights = zip(*FACTORIES)
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    out_file = out_dir / "augmented.jsonl"
    uid_start = 1_000_000

    with open(out_file, "w", encoding="utf-8") as fh:
        for i in range(args.rows):
            factory = rng.choices(factory_names, weights=probs, k=1)[0]
            text, spans = factory(rng)

            # Deduplicate and sort spans by start.
            seen: set[tuple[int, int, str]] = set()
            deduped: list[dict] = []
            for sp in sorted(spans, key=lambda s: s["start"]):
                key = (sp["start"], sp["end"], sp["label"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(sp)

            row = {
                "source_text": text,
                "language": "en",
                "locale": "US",
                "split": "train",
                "privacy_mask": deduped,
                "uid": uid_start + i,
                "masked_text": "",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓  {args.rows} synthetic rows → {out_file}")


if __name__ == "__main__":
    main()
