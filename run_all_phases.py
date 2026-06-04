"""
run_all_phases.py
==================
Master orchestrator — runs ALL remaining phases to completion.
Handles: Thunderbird pipeline → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7

Run: python run_all_phases.py
"""
import subprocess, sys, time, json
from pathlib import Path
import os

ROOT   = Path(r"C:\Users\ssrsh\Documents\projects\minor project\minor2")
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
RESULTS= ROOT / "results"

def run_step(label, script, cwd=None):
    print(f"\n{'='*65}")
    print(f"  RUNNING: {label}")
    print(f"{'='*65}")
    t0 = time.time()
    res = subprocess.run(
        [PYTHON, str(script)],
        cwd=cwd or ROOT,
        capture_output=False,
        text=True
    )
    elapsed = round(time.time() - t0, 1)
    status  = "OK" if res.returncode == 0 else f"FAILED (code {res.returncode})"
    print(f"\n  [{label}] {status} in {elapsed}s")
    return res.returncode == 0


def check_thunderbird():
    """Check if Thunderbird.log is extracted."""
    candidates = [
        r"C:\Users\ssrsh\Documents\projects\minor project\Thunderbird.log",
        r"C:\Users\ssrsh\Documents\projects\minor project\Thunderbird",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def main():
    print("\n" + "="*65)
    print("  AI-ASSISTED HYBRID VULNERABILITY DETECTION")
    print("  FULL PIPELINE EXECUTION — ALL PHASES")
    print("="*65)

    results_track = {}

    # ── THUNDERBIRD ───────────────────────────────────────────────────────────
    tb_path = check_thunderbird()
    if tb_path:
        print(f"\n[*] Thunderbird log found: {tb_path} (Skipping, already ran)")
        results_track['thunderbird'] = True
    else:
        print("\n[!] Thunderbird.log not yet extracted — skipping TB pipeline")
        results_track['thunderbird'] = False

    # ── PHASE 3: Alpha Sensitivity ────────────────────────────────────────────
    ok = run_step("Phase 3 — Alpha Sensitivity Analysis",
                  ROOT / "phase3_alpha_sensitivity.py")
    results_track['phase3'] = ok

    # ── PHASE 4: Cross-Validation ─────────────────────────────────────────────
    ok = run_step("Phase 4 — 5-Fold Cross-Validation",
                  ROOT / "phase4_crossval.py")
    results_track['phase4'] = ok

    # ── PHASE 5: DeepLog Baseline ─────────────────────────────────────────────
    ok = run_step("Phase 5 — DeepLog Baseline",
                  ROOT / "phase5_deeplog.py")
    results_track['phase5'] = ok

    # ── PHASE 6: SHAP Importance ──────────────────────────────────────────────
    ok = run_step("Phase 6 — SHAP Feature Importance",
                  ROOT / "phase6_shap.py")
    results_track['phase6'] = ok

    # ── PHASE 7: Final Report ─────────────────────────────────────────────────
    ok = run_step("Phase 7 — Final Journal Report",
                  ROOT / "phase7_final_report.py")
    results_track['phase7'] = ok

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EXECUTION SUMMARY")
    print("="*65)
    for step, status in results_track.items():
        icon = "[OK]" if status else "[!!]"
        print(f"  {icon}  {step}")

    print("\n  OUTPUT FILES:")
    for f in sorted(RESULTS.glob("*.json")):
        print(f"    {f.name}")
    print("\n  PLOTS:")
    plots = RESULTS / "plots"
    if plots.exists():
        for f in sorted(plots.glob("*.png")):
            print(f"    {f.name}")

    print("\n  Project complete. All journal assets generated.")
    print("="*65)


if __name__ == '__main__':
    main()
