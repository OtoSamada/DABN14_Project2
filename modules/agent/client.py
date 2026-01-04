import json
import pandas as pd
from typing import Dict, Any, List
from cerebras.cloud.sdk import Cerebras


def prepare_llm_payload(row: pd.Series) -> Dict[str, Any]:
    """
    Transforms a raw dataframe row into a clean, human-readable dictionary for the LLM.
    
    Args:
        row: A pandas Series containing customer data
        
    Returns:
        Dictionary with formatted customer information for LLM consumption
    """
    has_loan_val = row.get('has_loan', 0)
    if isinstance(has_loan_val, str):
        has_loan_val = 1 if has_loan_val.strip().lower() == 'yes' else 0

    # Base Essentials
    payload = {
        "customer_id": int(row.get('customer_id', -1)),
        "income": f"${row.get('monthly_income_usd', 0):.0f}",
        "expenses": f"${row.get('monthly_expenses_usd', 0):.0f}",
        "savings": f"${row.get('savings_usd', 0):.0f}",
        "credit_score": int(row.get('credit_score', 600)),
        "borrowing_power": f"${row.get('borrowing_power_usd', 0):.0f}",
        "responsiveness": f"{row.get('responsiveness', 0.5):.2f}",
        "has_loan": int(has_loan_val),
        "has_deposit": int(row.get('has_deposit', 0))
    }
    
    # Conditional Context (Only add if relevant to save tokens)
    if row.get('has_loan', 0) == 1:
        payload["current_loan"] = {
            "type": row.get('loan_type', 'Unspecified'),
            "amount": f"${row.get('loan_amount_usd', 0):.0f}",
            "rate": f"{row.get('loan_interest_rate_pct', 0):.1f}%"
        }
        
    if row.get('has_deposit', 0) == 1:
        payload["current_savings_rate"] = f"{row.get('deposit_interest_rate', 0):.2f}%"
        
    return payload


def batch_process_customers(
    customers: List[Dict[str, Any]], 
    product_key: str,
    product_catalog: Dict,
    api_key: str,
    model: str = "gpt-oss-120b",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    use_system_prompt: bool = True
) -> pd.DataFrame:
    """
    Processes customers using Cerebras LLM for product evaluation.
    Handles batches via a single prompt with clear JSON output instructions.
    
    Args:
        customers: List of customer payload dictionaries
        product_key: Key of the product to evaluate from catalog
        product_catalog: Dictionary or dict-like object containing Product instances
        api_key: Cerebras API key
        model: Model name to use (default: "gpt-oss-120b")
        temperature: Sampling temperature (default: 0.1)
        max_tokens: Maximum tokens in response (default: 4096)
        use_system_prompt: Whether to use system/user split (default: False)
        
    Returns:
        DataFrame with columns: customer_id, decision, satisfaction_score, reason
    """
    # Initialize Cerebras client
    client = Cerebras(api_key=api_key)
    
    # 1. Prepare Product Context - Compatible with both dict and dataclass
    product_offerings = "\n".join([
        f"- {(prod.name if hasattr(prod, 'name') else prod['name'])}: "
        f"{(prod.description if hasattr(prod, 'description') else prod['description'])}"
        for name, prod in product_catalog.items()
    ])
    
    # Get product details (works with both Product dataclass and dict)
    product = product_catalog.get(product_key)
    if product is None:
        raise ValueError(f"Product '{product_key}' not found in catalog")
    
    product_name = product.name if hasattr(product, 'name') else product['name']
    product_desc = product.description if hasattr(product, 'description') else product['description']
    product_message = product.message if hasattr(product, 'message') else product['message']
    
    # 2. Clean Data (remove customer_id for prompt)
    clean_profiles = [{k: v for k, v in c.items() if k != 'customer_id'} for c in customers]
    current_batch_size = len(clean_profiles)
    
    # 3. Build Prompt Components
    system_context = f"""
        ### CONTEXT:
        You are a simulator for mass retail bank customers.
        Evaluate offers based strictly on the provided profiles.

        ### CRITICAL SIMULATION RULES:
        1. **PRE-APPROVAL OVERRIDE:** All offers are GUARANTEED PRE-APPROVED. The bank has waived all eligibility requirements (Credit Score, Income, etc.).
        2. **DO NOT REJECT** based on ineligibility. The offer is valid regardless of profile stats.
        3. **EVALUATE VALUE ONLY:** Decide based on: "Does this make my financial life better?" (e.g., lower rate, cash back, fee waiver).

        PRODUCT OFFERINGS:
        {product_offerings}

        ### TASK:
        For each customer (in the exact input order), output a JSON array of {current_batch_size} objects with:
        - "decision": "accepted" or "rejected"
        - "satisfaction_score": integer from 1 to 10
        - "reason": string (max 10 words, focused on VALUE/NEED, not eligibility)

        ### OUTPUT FORMAT:
        Respond ONLY with a VALID JSON ARRAY. No other text. Example:
        [{{"decision": "accepted", "satisfaction_score": 8, "reason": "Lower interest saves me money"}}, ...]
    """.strip()

    user_content = f"""
        ### INPUT DATA:
        CUSTOMER PROFILES (in order):
        {json.dumps(clean_profiles)}

        OFFER TO EVALUATE:
        Message: "{product_message}"
        Product: "{product_name}"
        Description: "{product_desc}"
        """.strip()

    # 4. Build messages array based on model support
    if use_system_prompt:
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_content}
        ]
    else:
        # Combine into single user message (more compatible)
        combined_prompt = f"{system_context}\n\n{user_content}"
        messages = [{"role": "user", "content": combined_prompt}]

    try:
        # 5. Call Cerebras
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        output_text = response.choices[0].message.content.strip()
        
        # 6. Parse JSON
        # Remove potential markdown code block wrappers
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]
        
        results = json.loads(output_text.strip())
        
        # Ensure it's a list
        if not isinstance(results, list):
            raise ValueError("Model did not return a JSON array")

        # 7. Re-attach customer IDs
        final_data = []
        for i, res in enumerate(results):
            if i < len(customers):
                cust_id = customers[i].get('customer_id')
                final_data.append({**res, 'customer_id': cust_id})
                
        return pd.DataFrame(final_data)

    except Exception as e:
        print(f"❌ Cerebras Error: {e}")
        print(f"Raw model output:\n{output_text if 'output_text' in locals() else 'N/A'}")
        return pd.DataFrame()