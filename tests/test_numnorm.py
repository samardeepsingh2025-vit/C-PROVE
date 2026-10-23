import pytest
from cprove.numnorm import extract_values, numeric_report


@pytest.mark.parametrize("text,vals", [
    ("twenty five", [25]), ("twenty-five euros", [25]), ("25", [25]), ("two hundred and fifty", [250]), ("a hundred", [100]),
    ("twelve point five euros", [12.5]), ("12.5 euros", [12.5]), ("1,200", [1200]), ("5k", [5000]), ("2.5 million", [2.5e6]),
    ("three thousand five hundred", [3500]), ("forty five percent", [45]), ("25%", [25]), ("twenty one", [21]),
    ("fifteen twenty", [15, 20]), ("two thousand and twenty", [2020])])
def test_extract(text, vals):
    assert sorted(extract_values(text)) == sorted(float(v) for v in vals)


TR = "the price will be twenty five euros and we need fifteen buttons. Maybe 12.5 for the cost, around 1,200 units"


@pytest.mark.parametrize("claim,unseen", [
    ("The price was set at 25 euros.", []), ("The price was set at 30 euros.", [30.0]),
    ("The cost target is twelve point five euros", []), ("One of the designers proposed 15 buttons", []),
    ("about 1200 units", []), ("40 people attended", [40.0]), ("No numbers here", [])])
def test_report(claim, unseen):
    assert numeric_report(TR, claim)["unseen"] == unseen


def test_bare_one_not_a_number_on_claim_side():
    assert extract_values("one of the designers", claim_side=True) == []
    assert extract_values("one euro", claim_side=True) == [1.0]


def test_spoken_money_forms():
    from cprove.numnorm import numeric_report
    for spoken in ("twelve fifty euros", "twelve euros fifty", "twelve and a half euros", "12 euros 50"):
        assert numeric_report(f"the cap is {spoken} per unit", "The budget was 12.50 Euros per unit.", False)["unseen"] == []
    assert numeric_report("the cap is twelve euros", "The budget was 12.50 Euros per unit.", False)["unseen"] == [12.5]
