from dataclasses import dataclass
from typing import List


@dataclass
class Product:
    """Represents a banking product offering."""
    
    name: str
    target_cluster: List[int]
    description: str
    message: str
    
    def __post_init__(self):
        """Validate product data after initialization."""
        if not self.name:
            raise ValueError("Product name cannot be empty")
        if not isinstance(self.target_cluster, list):
            raise ValueError("target_cluster must be a list")
        if not self.target_cluster:
            raise ValueError("Product must target at least one cluster")


# Product catalog using dataclasses
PRODUCT_CATALOG = {
    # --- PAIR 1: PRIME POWER USERS (High Income, High Score) ---
    "Apex_Rewards_Signature": Product(
        name="Apex Rewards Signature",
        target_cluster=[5],
        description=(
            "**The Card That Pays You Back.**\n\n"
            "Stop settling for 1% points. The Apex Signature card is designed for high-volume spenders who want simplicity and speed. "
            "Earn a flat 3% cash back on every single purchase—no rotating categories, no activation required.\n\n"
            "**Key Features:**\n"
            "• **3% Unlimited Cash Back** on all transactions.\n"
            "• **$15,000 Starting Limit** to match your lifestyle.\n"
            "• **No Foreign Transaction Fees** for global travel.\n"
            "• **Instant Digital Issuance:** Add to Apple/Google Pay immediately."
        ),
        message=(
            "You're leaving money on the table with every swipe. "
            "Upgrade to the card that pays you 3% on everything—your spending habits have already earned it."
        )
    ),
    
    "Apex_Founders_Edition": Product(
        name="Apex Founders Edition",
        target_cluster=[3],
        description=(
            "**The Ultimate Financial Shield.**\n\n"
            "Exclusive to our pre-approved clients, the Founders Edition gives you breathing room to optimize your cash flow. "
            "Move high-interest balances from other banks to us and pay absolutely zero interest for the next 18 months.\n\n"
            "**Key Features:**\n"
            "• **0% APR for 18 Months** on Balance Transfers & New Purchases.\n"
            "• **$0 Transfer Fee** (Waived for first 30 days).\n"
            "• **3% Cash Back** after the promo period ends.\n"
            "• **Pre-Approved Status:** No credit check required to activate."
        ),
        message=(
            "We've pre-approved you for the Founders Edition. "
            "Move your existing high-rate balances to us and pay 0% interest until 2027. "
            "No application required—just accept to activate."
        )
    ),

    # --- PAIR 2: DISTRESSED BORROWERS (Maxed Debt, Low Score) ---
    "Unified_Balance_Loan": Product(
        name="Unified Balance Loan",
        target_cluster=[2],
        description=(
            "**One Payment. Lower Rate. Zero Stress.**\n\n"
            "Juggling multiple due dates is expensive and stressful. The Unified Balance Loan combines your credit cards and personal loans "
            "into a single, fixed monthly payment that is lower than what you pay now.\n\n"
            "**Key Features:**\n"
            "• **Fixed Rate:** Your rate never increases, unlike credit cards.\n"
            "• **Cash Flow Boost:** Reduces your total monthly obligation by ~15%.\n"
            "• **End Date:** A clear timeline to be debt-free (e.g., 36 months).\n"
            "• **Direct Pay:** We send the money directly to your other creditors for you."
        ),
        message=(
            "Simplify your finances. Combine your scattered payments into one "
            "manageable monthly bill and lower your total monthly commitment immediately."
        )
    ),
    
    "Relief_Plus_Consolidation": Product(
        name="Relief+ Consolidation",
        target_cluster=[1],
        description=(
            "**Immediate Relief for Your Budget.**\n\n"
            "Get back on track without the upfront cost. We have designed the Relief+ program to put cash back in your pocket from Day 1. "
            "Consolidate your debt today, and we cover the costs.\n\n"
            "**Key Features:**\n"
            "• **$0 Origination Fee:** We waive the standard $300 setup fee.\n"
            "• **Skip-a-Payment:** Your first payment isn't due for 45 days.\n"
            "• **Rate Cap Guarantee:** Your new rate will be lower than your weighted average.\n"
            "• **Credit Score Repair:** Consistent on-time payments help rebuild your score fast."
        ),
        message=(
            "Stop the bleeding. We will consolidate your debt AND waive the $300 "
            "origination fee if you finalize today. Plus, make no payments for the first 30 days."
        )
    ),

    # --- PAIR 3: NEW PROSPECTS (Clean Slate) ---
    "FlexAccess_Line": Product(
        name="FlexAccess Line",
        target_cluster=[8],
        description=(
            "**Cash on Demand. Interest on Terms.**\n\n"
            "Life doesn't always align with payday. The FlexAccess Line is a reusable safety net of $2,000 that sits ready in your app. "
            "Transfer cash to your checking instantly, and only pay interest on the days you actually use it.\n\n"
            "**Key Features:**\n"
            "• **$2,000 Credit Limit:** Available 24/7.\n"
            "• **Pay for What You Use:** If you borrow $100 for 2 days, you pay pennies.\n"
            "• **Revolving:** Repay it and borrow it again immediately.\n"
            "• **Cheaper than Overdrafts:** Significantly lower rates than NSF fees."
        ),
        message=(
            "Liquid cash, on demand. Unlock a $2,000 credit line to handle "
            "life's surprises. Only pay for what you use."
        )
    ),
    
    "SafetyNet_Starter": Product(
        name="SafetyNet Starter",
        target_cluster=[9],
        description=(
            "**Banking Without the 'Gotcha'.**\n\n"
            "We believe banking should be free and forgiving. Open a SafetyNet Checking account today and enjoy a $500 buffer "
            "that covers you if your balance dips below zero—interest-free.\n\n"
            "**Key Features:**\n"
            "• **No Monthly Fees:** Keeps more money in your pocket.\n"
            "• **$500 Interest-Free Overdraft:** Valid for the first 90 days.\n"
            "• **Early Pay:** Get your paycheck up to 2 days early.\n"
            "• **No Minimum Balance:** Start with whatever you have."
        ),
        message=(
            "Banking without the 'gotcha' fees. Open a SafetyNet account and get "
            "$500 in fee-free overdraft protection. It's a safety net, not a debt trap."
        )
    ),

    # --- PAIR 4: SLEEPING SAVERS (High Deposits, Low Rates) ---
    "YieldMax_Account": Product(
        name="YieldMax Account",
        target_cluster=[10],
        description=(
            "**Make Your Money Work Harder.**\n\n"
            "Your current savings account is barely keeping up. Switch to YieldMax and earn 4.50% APY on every dollar. "
            "It is the exact same safety and liquidity you are used to, just with 100x the interest.\n\n"
            "**Key Features:**\n"
            "• **4.50% APY:** Variable rate pegged to the top of the market.\n"
            "• **Daily Compounding:** Watch your balance grow every morning.\n"
            "• **No Lock-up:** Withdraw your money anytime, instantly.\n"
            "• **FDIC Insured:** 100% safe up to $250k."
        ),
        message=(
            "Your money is sleeping. Wake it up with 4.5% APY. "
            "Takes 2 minutes to switch and start earning real returns."
        )
    ),
    
    "YieldMax_Accelerator": Product(
        name="YieldMax Accelerator",
        target_cluster=[12],
        description=(
            "**Exclusive Upgrade: Beat the Market Rate.**\n\n"
            "Because you are a valued client, we have unlocked a special 'Accelerator' tier for your account. "
            "Switch to YieldMax today and get a boosted 5.00% APY guaranteed for 6 months.\n\n"
            "**Key Features:**\n"
            "• **5.00% APY:** Includes a 0.50% bonus on top of our standard rate.\n"
            "• **Rate Lock:** Your bonus is guaranteed for 6 months.\n"
            "• **Instant Transfer:** Move your existing low-rate balance with one tap.\n"
            "• **Limited Time:** This offer expires in 48 hours."
        ),
        message=(
            "Urgent: You are earning near-zero interest. We've unlocked a 5.0% APY upgrade for your account valid for 48 hours."
        )
    ),
    
    "YieldMax_Premier": Product(
        name="YieldMax Premier Upgrade",
        target_cluster=[11],
        description=(
            "**You Have Earned Premier Status.**\n\n"
            "Your strong savings history qualifies you for our highest tier. The YieldMax Premier account is designed for serious savers "
            "who demand the absolute highest yield and concierge service.\n\n"
            "**Key Features:**\n"
            "• **5.10% APY:** The highest rate our bank offers publicly or privately.\n"
            "• **Fee-Free Wires:** Send domestic wires at no cost.\n"
            "• **Priority Support:** Direct line to our US-based premier team.\n"
            "• **Higher Limits:** Increased daily transfer and withdrawal limits."
        ),
        message=(
            "You qualify for Premier Status. Because of your strong savings history, we are upgrading you to our top tier: 5.10% APY + Zero Fees."
        )
    ),

    # --- PAIR 5: YIELD OPTIMIZERS (Maxed Rates) ---
    "WealthDirect_Index": Product(
        name="WealthDirect Index Account",
        target_cluster=[4],
        description=(
            "**Graduate from Savings to Wealth.**\n\n"
            "Savings accounts have a ceiling. To grow real wealth, you need the market. WealthDirect builds you a personalized, "
            "diversified portfolio of low-cost index funds based on your risk tolerance.\n\n"
            "**Key Features:**\n"
            "• **Automated Investing:** We rebalance your portfolio for you.\n"
            "• **Low Cost:** Only 0.25% annual advisory fee.\n"
            "• **Tax Optimized:** We automatically harvest losses to lower your tax bill.\n"
            "• **Global Diversification:** Own a slice of 10,000+ companies."
        ),
        message=(
            "You've hit the ceiling on savings rates. It's time to graduate. "
            "Turn your passive savings into active wealth with our automated index portfolios."
        )
    ),
    
    "SmartLink_Offset": Product(
        name="SmartLink Offset",
        target_cluster=[0],
        description=(
            "**Stop Paying Interest on Money You Have.**\n\n"
            "It doesn't make sense to pay 15% on a loan while earning 5% on savings. The SmartLink Offset connects your accounts "
            "so every dollar in savings temporarily 'pays off' your loan principal, slashing your interest bill.\n\n"
            "**Key Features:**\n"
            "• **Interest Offset:** $10k in savings cancels interest on $10k of debt.\n"
            "• **Keep Your Cash:** Your savings are NOT spent; they just sit there saving you money.\n"
            "• **Liquidity:** Withdraw your savings anytime if you need them.\n"
            "• **Massive Savings:** Effectively earn a 15%+ return (your loan rate) on your cash."
        ),
        message=(
            "Stop paying interest on money you already have. "
            "Link your savings to your loan and slash your interest payments to near zero—without spending your savings."
        )
    ),

    # --- PAIR 6: COLLATERALIZED BORROWERS (Loan + Deposit) ---
    "Secured_Rate_Smasher": Product(
        name="Secured Rate Smasher",
        target_cluster=[7],
        description=(
            "**Leverage Your Assets. Smash Your Rate.**\n\n"
            "Why pay unsecured rates when you have secured assets? Pledging your savings account as collateral for your loan "
            "drops your risk to zero—and we pass the savings directly to you.\n\n"
            "**Key Features:**\n"
            "• **Rate Drop:** instantly lower your APR by 5-8%.\n"
            "• **Keep Growing:** Your deposit continues to earn interest while locked.\n"
            "• **Credit Boost:** Secured loans look great on your credit report.\n"
            "• **Flexible Release:** Funds are released as you pay down the principal."
        ),
        message=(
            "Use your savings to save your credit. Secure your loan with your deposit "
            "to drop your rate by 5% instantly. Your savings keep growing while your payments shrink."
        )
    ),
    
    "CashBack_Refinance": Product(
        name="Cash-Back Refinance",
        target_cluster=[6],
        description=(
            "**Lower Rates + Cash in Your Pocket.**\n\n"
            "Refinancing shouldn't just save you money later; it should pay you now. Secure your existing loan with your savings balance "
            "to lower your rate AND get an instant cash credit.\n\n"
            "**Key Features:**\n"
            "• **$100 Cash Bonus:** Credited to your account immediately upon closing.\n"
            "• **5-8% Rate Reduction:** Save hundreds in interest per year.\n"
            "• **No Credit Check:** Your savings are the only approval we need.\n"
            "• **Win-Win:** Lower payments + Instant Cash."
        ),
        message=(
            "Unlock a lower rate + $100 cash. Secure your loan with your savings today, "
            "and we'll credit $100 to your account instantly. The cash is yours now."
        )
    )
}

def get_product(product_key: str) -> Product:
    """Safely retrieve a product from the catalog."""
    if product_key not in PRODUCT_CATALOG:
        raise KeyError(f"Product '{product_key}' not found in catalog")
    return PRODUCT_CATALOG[product_key]


def get_products_for_cluster(cluster_id: int) -> List[Product]:
    """Get all products targeting a specific cluster."""
    return [
        product for product in PRODUCT_CATALOG.values()
        if cluster_id in product.target_cluster
    ]