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


PRODUCT_CATALOG = {
    # --- PAIR 1: LEVERAGED BORROWERS (Debt + Cash) ---
    # Strategy: "Swap bad debt for good debt using their asset."
    
    # OLD: Secured_Rate_Smasher
    "Priority_Secured_Refinance": {
        "name": "Priority Secured Refinance",
        "target_cluster": [0],  # High Response
        "description": (
            "**Leverage Your Portfolio. Lower Your Rate.**\n\n"
            "A strategic lending solution that allows you to pledge your existing savings as collateral. "
            "By securing your loan, you significantly reduce your risk profile, allowing us to offer you our lowest possible APR.\n\n"
            "**Key Features:**\n"
            "• **Preferential Rates:** APR reduction of 5-8% vs. unsecured loans.\n"
            "• **Asset Growth:** Your pledged deposit continues to accrue interest.\n"
            "• **Credit Building:** Secured installment loans positively impact credit history.\n"
            "• **Flexible Terms:** 12 to 60-month repayment options."
        ),
        "message": (
            "Use your savings to optimize your debt. Secure your loan with your deposit "
            "to drop your rate by 5% instantly. Your savings keep growing while your payments shrink."
        )
    },
    
    # OLD: CashBack_Refinance
    "Priority_Secured_Plus": {
        "name": "Priority Secured Refinance (Cash Offer)",
        "target_cluster": [1],  # Low Response (Needs Cash Nudge)
        "description": (
            "**Rate Reduction + Immediate Cash Incentive.**\n\n"
            "An exclusive offer for valued depositors. Refinance your unsecured debt into a secured facility to lower your monthly interest, "
            "plus receive an immediate statement credit upon closing.\n\n"
            "**Key Features:**\n"
            "• **$100 Closing Bonus:** Credited to your account immediately.\n"
            "• **Significant Savings:** 5-8% average rate reduction.\n"
            "• **Streamlined Approval:** Expedited processing based on your deposit relationship.\n"
            "• **Liquidity Management:** Lower monthly payments improve cash flow."
        ),
        "message": (
            "Unlock a lower rate + $100 cash. Secure your loan with your savings today, "
            "and we'll credit $100 to your account instantly."
        )
    },

    # --- PAIR 2: PURE BORROWERS (Debt + No Cash) ---
    # Strategy: "Lower the monthly bleed."
    
    # OLD: Unified_Balance_Loan
    "Streamline_Consolidation_Loan": {
        "name": "Streamline Consolidation Loan",
        "target_cluster": [6],  # High Response
        "description": (
            "**Simplify Your Finances. One Payment, Fixed Rate.**\n\n"
            "Eliminate the complexity of multiple high-interest obligations. The Streamline Consolidation Loan combines your "
            "outstanding balances into a single, predictable monthly payment with a competitive fixed rate.\n\n"
            "**Key Features:**\n"
            "• **Fixed APR:** Protection against rising market rates.\n"
            "• **Payment Reduction:** Lower your total monthly cash outflow by up to 15%.\n"
            "• **Structured Repayment:** A clear 36 or 48-month path to becoming debt-free.\n"
            "• **Direct Creditor Pay:** We handle the payoffs for you."
        ),
        "message": (
            "Simplify your finances. Combine your scattered payments into one "
            "manageable monthly bill and lower your total monthly commitment immediately."
        )
    },
    
    # OLD: Relief_Plus_Consolidation
    "Essential_Relief_Loan": {
        "name": "Essential Relief Loan",
        "target_cluster": [3],  # Low Response (Needs Fee Waiver)
        "description": (
            "**Budget Relief with Zero Upfront Costs.**\n\n"
            "Designed to provide immediate liquidity support. We have waived all origination fees to ensure "
            "every dollar borrowed goes toward reducing your principal balances.\n\n"
            "**Key Features:**\n"
            "• **Fee Waiver:** $0 Origination Fee (Standard: $300).\n"
            "• **Deferred First Payment:** No payment due for the first 45 days.\n"
            "• **Rate Guarantee:** Your new APR will be lower than your current weighted average.\n"
            "• **Credit Rehabilitation:** Consistent payments help restore your credit profile."
        ),
        "message": (
            "Stop the bleeding. We will consolidate your debt AND waive the $300 "
            "origination fee if you finalize today. Plus, make no payments for the first 30 days."
        )
    },

    # --- PAIR 3: PURE SAVERS (Cash + No Debt) ---
    # Strategy: "Cross-sell or Retain."
    
    # OLD: Apex_Rewards_Signature
    "Prestige_Cash_Rewards": {
        "name": "Prestige Cash Rewards Visa®",
        "target_cluster": [5],  # High Response
        "description": (
            "**Premium Returns on Every Purchase.**\n\n"
            "A flagship card for clients who value efficiency. Earn an industry-leading flat rate on all spend "
            "without the complexity of rotating categories or enrollment caps.\n\n"
            "**Key Features:**\n"
            "• **3% Unlimited Cash Back** on all transactions.\n"
            "• **High Credit Line:** Starting limits from $15,000.\n"
            "• **Travel Ready:** No Foreign Transaction Fees.\n"
            "• **Digital First:** Instant provisioning to digital wallets."
        ),
        "message": (
            "You’re leaving money on the table. Upgrade to the card that pays you 3% on everything—your spending habits have already earned it."
        )
    },
    
    # OLD: YieldMax_Accelerator
    "Preferred_Savings_Select": {
        "name": "Preferred Savings Select",
        "target_cluster": [2],  # Low Response (Needs Retention Boost)
        "description": (
            "**Exclusive Rate Upgrade for Relationship Clients.**\n\n"
            "We have unlocked a promotional interest tier for your account. This 'Select' status grants you a market-leading APY "
            "for 6 months to maximize the yield on your idle cash.\n\n"
            "**Key Features:**\n"
            "• **5.00% Promotional APY:** Includes a 0.50% relationship bonus.\n"
            "• **Rate Assurance:** Promotional rate guaranteed for 6 months.\n"
            "• **Seamless Upgrade:** Convert your existing account instantly.\n"
            "• **Time-Sensitive:** Offer valid for 48 hours."
        ),
        "message": (
            "Urgent: You are earning near-zero interest. We've unlocked a 5.0% APY upgrade for your account valid for 48 hours."
        )
    },

    # --- PAIR 4: BLANK SLATES (No Cash / No Debt) ---
    # Strategy: "Acquisition & Trust."
    
    # OLD: FlexAccess_Line
    "Personal_Credit_Line": {
        "name": "Personal Credit Line",
        "target_cluster": [7],  # High Response
        "description": (
            "**On-Demand Liquidity. Revolving Access.**\n\n"
            "A flexible borrowing tool that works like a credit card but transfers cash directly to your checking account. "
            "Pay interest only on the funds you draw, for the days you use them.\n\n"
            "**Key Features:**\n"
            "• **$2,000 Revolving Limit:** Available 24/7 via mobile app.\n"
            "• **Per-Diem Interest:** No interest charged on unused balances.\n"
            "• **Auto-Replenish:** Credit becomes available again as you repay.\n"
            "• **Cost Efficient:** Rates significantly lower than overdraft fees."
        ),
        "message": (
            "Liquid cash, on demand. Unlock a $2,000 credit line to handle "
            "life’s surprises. Only pay for what you use."
        )
    },
    
    # OLD: SafetyNet_Starter
    "Confidence_Checking": {
        "name": "Confidence Checking",
        "target_cluster": [4],  # Low Response
        "description": (
            "**Banking with Built-In Protection.**\n\n"
            "A modern checking account designed to eliminate surprise fees. Includes a complimentary overdraft buffer "
            "to ensure declined transactions and NSF fees are a thing of the past.\n\n"
            "**Key Features:**\n"
            "• **No Monthly Maintenance Fees.**\n"
            "• **$500 Fee-Free Buffer:** Interest-free overdraft coverage (First 90 Days).\n"
            "• **Early Direct Deposit:** Access payroll funds up to 2 days early.\n"
            "• **No Minimum Balance Requirement.**"
        ),
        "message": (
            "Banking without the 'gotcha' fees. Open a Confidence Checking account and get "
            "$500 in fee-free overdraft protection."
        )
    }
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