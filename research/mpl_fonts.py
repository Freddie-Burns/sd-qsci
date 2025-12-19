from pathlib import Path
import matplotlib.font_manager as fm

font_path = Path(__file__).parent / "InterVariable.ttf"
fm.fontManager.addfont(str(font_path))