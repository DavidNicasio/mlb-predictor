import sys
from pathlib import Path

# Añadir src/ al sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import metrics


def test_fip_formula():
    # Caso 1: Abridor con 18 HR, 40 BB, 5 HBP, 180 K, 540 outs (180.0 IP) y fip_constant = 3.20
    # IP = 540 / 3 = 180.0
    # core = (13*18 + 3*(40+5) - 2*180) / 180.0 = (234 + 135 - 360) / 180.0 = 9 / 180.0 = 0.05
    # FIP = 0.05 + 3.20 = 3.25
    res = metrics.fip(hr=18, bb=40, hbp=5, k=180, outs=540, fip_constant=3.20)
    assert res == 3.25

    # Caso 2: Abridor dominante con 5 HR, 20 BB, 2 HBP, 100 K, 300 outs (100.0 IP) y fip_constant = 3.10
    # core = (13*5 + 3*22 - 200) / 100 = (65 + 66 - 200) / 100 = -69 / 100 = -0.69
    # FIP = -0.69 + 3.10 = 2.41
    res2 = metrics.fip(hr=5, bb=20, hbp=2, k=100, outs=300, fip_constant=3.10)
    assert res2 == 2.41


def test_woba_formula():
    # Caso: 50 BB, 3 IBB, 5 HBP, 90 S1, 25 S2, 2 S3, 20 HR, 520 AB, 4 SF en temporada 2024
    # uBB = 50 - 3 = 47
    # Denom = 520 + 47 + 4 + 5 = 576
    # Num = 0.689*47 + 0.720*5 + 0.883*90 + 1.257*25 + 1.593*2 + 2.058*20
    # Num = 32.383 + 3.6 + 79.47 + 31.425 + 3.186 + 41.16 = 191.224
    # wOBA = 191.224 / 576 = 0.331986 -> 0.332
    res = metrics.woba(bb=50, ibb=3, hbp=5, singles=90, doubles=25, triples=2, hr=20, ab=520, sf=4, season=2024)
    assert res == 0.332


def test_shrink_rate():
    # Caso observada=0.400, muestra=100, liga=0.320, k=100
    # weight = 100 / (100+100) = 0.50
    # res = 0.50*0.400 + 0.50*0.320 = 0.360
    res = metrics.shrink_rate(observed_rate=0.400, sample_size=100, league_rate=0.320, k=100)
    assert res == 0.3600
