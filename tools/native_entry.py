"""Source-checkout entry point; installed packages can use -m nanoai.native_worker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.native_worker import main
main()
