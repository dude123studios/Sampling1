"""
Mechanistic Interpretability Experiments Runner

Orchestrates the full mechanistic interpretability pipeline:
1. Token Impact Identification: Find branching points where token choice matters
2. Direct Logit Attribution (DLA): Quantify layer-wise contributions
3. Activation Patching: Identify critical decision layers via causal intervention
4. Gradient Attribution: Measure sensitivity/instability at each layer

Usage:
  python mech_interp/run_mech_interp.py --stage all
  python mech_interp/run_mech_interp.py --stage token_impact
  python mech_interp/run_mech_interp.py --stage dla
  python mech_interp/run_mech_interp.py --stage patching
  python mech_interp/run_mech_interp.py --stage gradient
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class MechInterpRunner:
    def __init__(
        self,
        token_impact_config: str = "configs/mech_interp/token_impact_config.yaml",
        dla_patching_config: str = "configs/mech_interp/dla_patching_config.yaml",
        skip_existing: bool = False
    ):
        """Initialize the mechanistic interpretability runner.

        Args:
            token_impact_config: Path to token impact config
            dla_patching_config: Path to DLA/patching/gradient config
            skip_existing: Skip stages if output already exists
        """
        self.token_impact_config = token_impact_config
        self.dla_patching_config = dla_patching_config
        self.skip_existing = skip_existing

        # Expected output paths
        self.token_impact_results = Path("mech_interp/token_impact_results/token_impact_results.json")
        self.dla_results = Path("mech_interp/dla_results/dla_results.json")
        self.patching_results = Path("mech_interp/patching_results/patching_results.json")
        self.gradient_results = Path("mech_interp/gradient_results/gradient_results.json")

    def run_command(self, cmd: list, description: str):
        """Run a command and handle errors."""
        log.info(f"Running: {description}")
        log.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            log.info(f"✓ {description} completed successfully")
            if result.stdout:
                log.info(f"Output:\n{result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            log.error(f"✗ {description} failed")
            log.error(f"Error:\n{e.stderr}")
            return False

    def run_token_impact(self):
        """Run token impact identification."""
        if self.skip_existing and self.token_impact_results.exists():
            log.info(f"Skipping token impact (output exists: {self.token_impact_results})")
            return True

        log.info("=" * 80)
        log.info("STAGE 1: Token Impact Identification")
        log.info("=" * 80)
        log.info("This identifies branching points where token choice creates divergent paths")
        log.info("")

        cmd = [
            sys.executable,
            "mech_interp/token_impact.py",
            "--config", self.token_impact_config
        ]

        return self.run_command(cmd, "Token Impact Identification")

    def run_dla(self):
        """Run Direct Logit Attribution analysis."""
        if not self.token_impact_results.exists():
            log.error("Cannot run DLA: token impact results not found")
            log.error(f"Expected: {self.token_impact_results}")
            log.error("Run token_impact stage first")
            return False

        if self.skip_existing and self.dla_results.exists():
            log.info(f"Skipping DLA (output exists: {self.dla_results})")
            return True

        log.info("=" * 80)
        log.info("STAGE 2: Direct Logit Attribution (DLA)")
        log.info("=" * 80)
        log.info("This quantifies which layers contribute to preferring token1 over token2")
        log.info("")

        cmd = [
            sys.executable,
            "mech_interp/dla.py",
            "--config", self.dla_patching_config,
            "--branching-points", str(self.token_impact_results)
        ]

        return self.run_command(cmd, "Direct Logit Attribution")

    def run_patching(self):
        """Run activation patching analysis."""
        if not self.token_impact_results.exists():
            log.error("Cannot run patching: token impact results not found")
            log.error(f"Expected: {self.token_impact_results}")
            log.error("Run token_impact stage first")
            return False

        if self.skip_existing and self.patching_results.exists():
            log.info(f"Skipping patching (output exists: {self.patching_results})")
            return True

        log.info("=" * 80)
        log.info("STAGE 3: Activation Patching")
        log.info("=" * 80)
        log.info("This identifies critical layers via causal intervention")
        log.info("")

        cmd = [
            sys.executable,
            "mech_interp/patching.py",
            "--config", self.dla_patching_config,
            "--branching-points", str(self.token_impact_results)
        ]

        return self.run_command(cmd, "Activation Patching")

    def run_gradient(self):
        """Run gradient attribution analysis."""
        if not self.token_impact_results.exists():
            log.error("Cannot run gradient: token impact results not found")
            log.error(f"Expected: {self.token_impact_results}")
            log.error("Run token_impact stage first")
            return False

        if self.skip_existing and self.gradient_results.exists():
            log.info(f"Skipping gradient (output exists: {self.gradient_results})")
            return True

        log.info("=" * 80)
        log.info("STAGE 4: Gradient Attribution")
        log.info("=" * 80)
        log.info("This measures sensitivity/instability at each layer")
        log.info("")

        cmd = [
            sys.executable,
            "mech_interp/gradient.py",
            "--config", self.dla_patching_config,
            "--branching-points", str(self.token_impact_results)
        ]

        return self.run_command(cmd, "Gradient Attribution")

    def run_all(self):
        """Run all stages in sequence."""
        log.info("=" * 80)
        log.info("MECHANISTIC INTERPRETABILITY PIPELINE")
        log.info("=" * 80)
        log.info("")

        stages = [
            ("token_impact", self.run_token_impact),
            ("dla", self.run_dla),
            ("patching", self.run_patching),
            ("gradient", self.run_gradient)
        ]

        results = {}
        for stage_name, stage_fn in stages:
            log.info("")
            success = stage_fn()
            results[stage_name] = success

            if not success:
                log.error(f"Stage '{stage_name}' failed. Stopping pipeline.")
                break

        # Summary
        log.info("")
        log.info("=" * 80)
        log.info("PIPELINE SUMMARY")
        log.info("=" * 80)
        for stage_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            log.info(f"{stage_name:20s}: {status}")

        all_success = all(results.values())
        if all_success:
            log.info("")
            log.info("All stages completed successfully!")
            log.info("")
            log.info("Results:")
            log.info(f"  Token Impact:  {self.token_impact_results}")
            log.info(f"  DLA:           {self.dla_results}")
            log.info(f"  Patching:      {self.patching_results}")
            log.info(f"  Gradient:      {self.gradient_results}")

        return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Run Mechanistic Interpretability Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python mech_interp/run_mech_interp.py --stage all

  # Run individual stages
  python mech_interp/run_mech_interp.py --stage token_impact
  python mech_interp/run_mech_interp.py --stage dla
  python mech_interp/run_mech_interp.py --stage patching
  python mech_interp/run_mech_interp.py --stage gradient

  # Skip stages with existing results
  python mech_interp/run_mech_interp.py --stage all --skip-existing
        """
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "token_impact", "dla", "patching", "gradient"],
        default="all",
        help="Which stage to run"
    )

    parser.add_argument(
        "--token-impact-config",
        type=str,
        default="configs/mech_interp/token_impact_config.yaml",
        help="Path to token impact config"
    )

    parser.add_argument(
        "--dla-patching-config",
        type=str,
        default="configs/mech_interp/dla_patching_config.yaml",
        help="Path to DLA/patching/gradient config"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages if output already exists"
    )

    args = parser.parse_args()

    # Create runner
    runner = MechInterpRunner(
        token_impact_config=args.token_impact_config,
        dla_patching_config=args.dla_patching_config,
        skip_existing=args.skip_existing
    )

    # Run requested stage
    if args.stage == "all":
        success = runner.run_all()
    elif args.stage == "token_impact":
        success = runner.run_token_impact()
    elif args.stage == "dla":
        success = runner.run_dla()
    elif args.stage == "patching":
        success = runner.run_patching()
    elif args.stage == "gradient":
        success = runner.run_gradient()
    else:
        log.error(f"Unknown stage: {args.stage}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
