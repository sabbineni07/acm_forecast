# Required Columns for ACM Forecast Framework

## Analysis Summary

Based on code review of the ACM Forecast framework, the following columns are **actually used** in the codebase:

## Required Columns (Core - Always Needed)

These columns are directly referenced in the code and are essential for the framework to function:

1. **`usage_date`** (DATE)
   - **Config mapping**: `feature.date_column` (default: "UsageDateTime")
   - **Used in**: Data preparation, feature engineering, time series splitting, aggregation
   - **Purpose**: Time series index for forecasting
   - **From schema**: `usage_date` (DATE)

2. **`cost_in_billing_currency`** (DECIMAL(20,10))
   - **Config mapping**: `feature.target_column` (default: "PreTaxCost")
   - **Used in**: Target variable for all models, aggregation, feature engineering
   - **Purpose**: The cost value to forecast (target variable)
   - **From schema**: `cost_in_billing_currency` (DECIMAL(20,10))

3. **`quantity`** (DECIMAL(20,6))
   - **Used in**: Data preparation aggregation, derived feature calculation (cost per unit)
   - **Purpose**: Usage quantity for calculating derived metrics
   - **From schema**: `quantity` (DECIMAL(20,6))

4. **`meter_category`** (STRING)
   - **Used in**: Data segmentation, grouping in aggregations, categorical features for XGBoost
   - **Purpose**: Primary segmentation dimension
   - **From schema**: `meter_category` (STRING)

5. **`resource_location`** (STRING)
   - **Used in**: Data quality checks, grouping in aggregations, categorical features for XGBoost
   - **Purpose**: Regional segmentation
   - **From schema**: `resource_location` (STRING)

## Recommended Columns (Strongly Recommended)

These columns are used in specific features or quality checks:

6. **`subscription_id`** (STRING)
   - **Used in**: Data source mapping (optional filtering)
   - **Purpose**: Subscription-level analysis
   - **From schema**: `subscription_id` (STRING)

7. **`effective_price`** or **`unit_price`** (DECIMAL(20,10))
   - **Used in**: Data preparation aggregation (`ResourceRate`)
   - **Purpose**: Rate per unit calculation
   - **From schema**: `effective_price` (DECIMAL(20,10)) or `unit_price` (DECIMAL(20,10))
   - **Note**: Framework uses `ResourceRate` - can map from either field

8. **`service_tier`** (STRING)
   - **Used in**: Categorical feature for XGBoost, missing value handling
   - **Purpose**: Service tier categorization
   - **From schema**: `plan_name` (may need parsing) or derive from other fields
   - **Note**: Not directly in schema, may need to derive or use `plan_name`

9. **`billing_currency_code`** (STRING)
   - **Used in**: Data quality validation
   - **Purpose**: Currency validation
   - **From schema**: `billing_currency_code` (STRING)

## Optional Columns (Nice to Have)

These columns are referenced but not strictly required:

10. **`consumed_service`** (STRING)
    - **Used in**: Feature engineering (categorical features)
    - **From schema**: `consumed_service` (STRING)

11. **`meter_sub_category`** (STRING)
    - **Used in**: Can be useful for more granular segmentation
    - **From schema**: `meter_sub_category` (STRING)

12. **`resource_group`** (STRING)
    - **Used in**: Can be useful for resource-level analysis
    - **From schema**: `resource_group` (STRING)

## Complete Required Column List (Minimum Viable)

For the framework to work, you need at minimum these **5 columns**:

```python
REQUIRED_COLUMNS = [
    "usage_date",                    # DATE - Time series index
    "cost_in_billing_currency",      # DECIMAL(20,10) - Target variable
    "quantity",                      # DECIMAL(20,6) - For derived features
    "meter_category",                # STRING - Primary segmentation
    "resource_location",             # STRING - Regional segmentation
]
```

## Recommended Column List (Full Functionality)

For full framework functionality, use these **9 columns**:

```python
RECOMMENDED_COLUMNS = [
    "usage_date",                    # DATE
    "cost_in_billing_currency",      # DECIMAL(20,10)
    "quantity",                      # DECIMAL(20,6)
    "meter_category",                # STRING
    "resource_location",             # STRING
    "subscription_id",               # STRING
    "effective_price",               # DECIMAL(20,10) or unit_price
    "billing_currency_code",         # STRING
    "plan_name",                     # STRING (for service_tier derivation)
]
```

## Column Mapping (Old Schema → New Schema)

| Old Framework Name (PascalCase) | New Schema Name (snake_case) | Type | Required |
|--------------------------------|------------------------------|------|----------|
| UsageDateTime | usage_date | DATE | ✅ Yes |
| PreTaxCost | cost_in_billing_currency | DECIMAL(20,10) | ✅ Yes |
| UsageQuantity | quantity | DECIMAL(20,6) | ✅ Yes |
| MeterCategory | meter_category | STRING | ✅ Yes |
| ResourceLocation | resource_location | STRING | ✅ Yes |
| SubscriptionGuid | subscription_id | STRING | ⚠️ Recommended |
| ResourceRate | effective_price or unit_price | DECIMAL(20,10) | ⚠️ Recommended |
| Currency | billing_currency_code | STRING | ⚠️ Recommended |
| ServiceTier | plan_name (derive) | STRING | ⚠️ Recommended |

## Notes

1. **Date Field**: The schema uses `usage_date` (DATE) but the framework may expect timestamps. You may need to convert or the framework may handle DATE types directly.

2. **Service Tier**: The framework references `ServiceTier` but the schema doesn't have this directly. You can:
   - Derive from `plan_name` (parse Basic/Standard/Premium)
   - Leave as NULL if not critical
   - Use a default value

3. **Price Field**: Framework uses `ResourceRate` - you can use either `effective_price` or `unit_price` from the schema.

4. **Config Updates**: You may need to update `config.yaml` to use snake_case column names:
   ```yaml
   feature:
     target_column: "cost_in_billing_currency"
     date_column: "usage_date"
   ```


