HARAM_SECTORS = [
    "Banks",
    "Alcohol",
    "Gambling",
    "Tobacco",
    "Insurance",
    "Adult Entertainment",
    "Pork"
]


def check_sector(sector):
    sector_lower = sector.lower()

    # ครอบคลุมธุรกิจธนาคารและการเงินทั้งหมด → haram ทันที
    if "bank" in sector_lower or "financial" in sector_lower:
        return False, f"Core business is haram → company is haram ({sector})"

    if sector in HARAM_SECTORS:
        return False, f"Core business is haram → company is haram ({sector})"

    return True, "Core business is halal"


def check_financial_ratios(interest_ratio, debt_ratio, non_halal_ratio):
    reasons = []
    halal = True

    if interest_ratio >= 0.05:
        halal = False
        reasons.append("Interest income exceeds 5%")

    if debt_ratio >= 0.30:
        halal = False
        reasons.append("Debt exceeds 30% of total assets")

    if non_halal_ratio >= 0.05:
        halal = False
        reasons.append("Non-halal income exceeds 5%")

    if halal:
        reasons.append("All financial ratios are within halal thresholds")

    return halal, reasons