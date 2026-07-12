# Deterministic cron delivery

Normal Hermes cron jobs run a prompt or skills through an agent and retain an
output record. Trusted local scripts can instead use `execution_mode=script`
when the script itself is the complete renderer and no model judgment belongs
in the path.

```bash
hermes cron create '0 9 * * *' \
  --name 'Deterministic report' \
  --script report.py \
  --execution-mode script \
  --no-archive-output \
  --deduplicate-delivery \
  --deliver telegram
```

Script-mode jobs:

- run `~/.hermes/scripts/<name>.py` with the Hermes Python interpreter;
- do not create an `AIAgent`, model request, or session database record;
- pass bounded valid UTF-8 stdout as the message body, without cron wrappers,
  trimming, or `MEDIA:` interpretation; the selected platform may still apply
  its normal transport escaping;
- fail on missing, symlinked, non-owner-only, timed-out, oversized, non-UTF-8,
  empty, or nonzero-exit scripts;
- optionally skip the normal `~/.hermes/cron/output/` artifact with
  `--no-archive-output`;
- optionally deduplicate by SHA-256 of exact stdout. The delivery key advances
only after the target reports success, so failed sends remain retryable.
Failed one-shot script runs are rescheduled after a five-minute backoff instead
of being marked complete, and changing the delivery target clears the prior key.

Deduplication accepts one delivery target only; this avoids retrying a partially
successful multi-target send and duplicating the targets that already received it.

Script mode rejects prompts, skills, and model/provider overrides. Output is
limited to 1,800 UTF-16 units so it remains one message on supported chat
targets. Scripts are trusted executable code: keep `HERMES_HOME`, its `scripts`
tree, and the script owner-only, and make output safe for the configured
delivery target. Job metadata and the last successful
delivery hash remain in `~/.hermes/cron/jobs.json`; disabling the output archive
does not remove that operational state.

Existing jobs default to `execution_mode=agent`, output archival enabled, and
delivery deduplication disabled.
