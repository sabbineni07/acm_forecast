# Extracted Azure Cost Management Delta Table Schema (Snake Case)

## Complete Schema Definition

Based on the provided schema image, here is the extracted Azure ACM Delta table schema using snake_case naming convention:

```sql
CREATE TABLE azure_cost_management.amortized_costs (
    invoice_section_name STRING,
    account_name STRING,
    account_owner_id STRING,
    subscription_id STRING,
    subscription_name STRING,
    resource_group STRING,
    resource_location STRING,
    usage_date DATE,
    product_name STRING,
    meter_category STRING,
    meter_sub_category STRING,
    meter_id STRING,
    meter_name STRING,
    meter_region STRING,
    unit_of_measure STRING,
    quantity DECIMAL(20,6),
    effective_price DECIMAL(20,10),
    cost_in_billing_currency DECIMAL(20,10),
    cost_center STRING,
    consumed_service STRING,
    resource_id STRING,
    tags STRING,
    offer_id STRING,
    additional_info STRING,
    service_info1 STRING,
    service_info2 STRING,
    resource_name STRING,
    reservation_id STRING,
    reservation_name STRING,
    unit_price DECIMAL(20,10),
    product_order_id STRING,
    product_order_name STRING,
    term STRING,
    publisher_type STRING,
    publisher_name STRING,
    charge_type STRING,
    frequency STRING,
    pricing_model STRING,
    availability_zone STRING,
    billing_account_id BIGINT,
    billing_account_name STRING,
    billing_currency_code STRING,
    billing_period_start_date DATE,
    billing_period_end_date DATE,
    billing_profile_id BIGINT,
    billing_profile_name STRING,
    invoice_section_id INT,
    is_azure_credit_eligible STRING,
    part_number STRING,
    payg_price DECIMAL(20,10),
    plan_name STRING,
    service_family STRING,
    cost_allocation_rule_name STRING,
    benefit_id STRING,
    benefit_name STRING,
    _record_hash_sha256 STRING,
    input_file_name STRING,
    stg_upsert_ts TIMESTAMP
)
```

## Field Summary

**Total Fields: 57**

### Billing & Account Fields (11)
- invoice_section_name
- account_name
- account_owner_id
- billing_account_id
- billing_account_name
- billing_currency_code
- billing_period_start_date
- billing_period_end_date
- billing_profile_id
- billing_profile_name
- invoice_section_id

### Subscription & Resource Fields (8)
- subscription_id
- subscription_name
- resource_group
- resource_location
- resource_id
- resource_name
- availability_zone
- cost_center

### Usage & Meter Fields (9)
- usage_date
- meter_category
- meter_sub_category
- meter_id
- meter_name
- meter_region
- unit_of_measure
- quantity
- product_name

### Cost & Pricing Fields (6)
- cost_in_billing_currency
- effective_price
- unit_price
- payg_price
- pricing_model
- frequency

### Service & Product Fields (7)
- consumed_service
- service_info1
- service_info2
- service_family
- product_order_id
- product_order_name
- plan_name

### Reservation & Benefits (4)
- reservation_id
- reservation_name
- benefit_id
- benefit_name

### Additional Metadata (5)
- tags
- additional_info
- offer_id
- charge_type
- term

### Publisher Fields (3)
- publisher_type
- publisher_name
- part_number

### System Fields (3)
- is_azure_credit_eligible
- cost_allocation_rule_name
- input_file_name

### Audit/Technical Fields (2)
- _record_hash_sha256
- stg_upsert_ts

## Data Type Summary

- **STRING**: 47 fields
- **DECIMAL(20,6)**: 1 field (quantity)
- **DECIMAL(20,10)**: 4 fields (effective_price, cost_in_billing_currency, unit_price, payg_price)
- **DATE**: 3 fields (usage_date, billing_period_start_date, billing_period_end_date)
- **BIGINT**: 2 fields (billing_account_id, billing_profile_id)
- **INT**: 1 field (invoice_section_id)
- **TIMESTAMP**: 1 field (stg_upsert_ts)

## Key Differences from Previous Schema

1. **Naming Convention**: Uses snake_case instead of PascalCase
2. **Date Field**: `usage_date` (DATE) instead of `UsageDateTime` (TIMESTAMP)
3. **Cost Field**: `cost_in_billing_currency` instead of `PreTaxCost`
4. **Additional Fields**: Many more billing-related fields (billing_account_id, billing_profile_id, etc.)
5. **Reservation Fields**: Includes reservation_id and reservation_name
6. **Benefit Fields**: Includes benefit_id and benefit_name
7. **System Fields**: Includes _record_hash_sha256 and stg_upsert_ts for data lineage
8. **More Pricing Details**: effective_price, unit_price, payg_price instead of ResourceRate
9. **Product Fields**: product_order_id, product_order_name, plan_name
10. **Billing Period**: billing_period_start_date, billing_period_end_date

## Mapping to Framework Expected Fields

If the framework expects the old schema format, here's a mapping guide:

| Framework Expected | Actual Delta Table Field | Notes |
|-------------------|-------------------------|-------|
| SubscriptionGuid | subscription_id | Direct mapping |
| ResourceGroup | resource_group | Direct mapping |
| ResourceLocation | resource_location | Direct mapping |
| UsageDateTime | usage_date | Date type, may need conversion |
| MeterCategory | meter_category | Direct mapping |
| MeterSubCategory | meter_sub_category | Direct mapping |
| MeterId | meter_id | Direct mapping |
| MeterName | meter_name | Direct mapping |
| MeterRegion | meter_region | Direct mapping |
| UsageQuantity | quantity | Direct mapping |
| ResourceRate | effective_price or unit_price | May need to select one |
| PreTaxCost | cost_in_billing_currency | Direct mapping |
| ConsumedService | consumed_service | Direct mapping |
| ResourceType | (derived from resource_id) | May need parsing |
| InstanceId | resource_id | Direct mapping |
| Tags | tags | Direct mapping |
| OfferId | offer_id | Direct mapping |
| AdditionalInfo | additional_info | Direct mapping |
| ServiceInfo1 | service_info1 | Direct mapping |
| ServiceInfo2 | service_info2 | Direct mapping |
| ServiceName | product_name or consumed_service | May need selection |
| ServiceTier | (derived from plan_name) | May need parsing |
| Currency | billing_currency_code | Direct mapping |
| UnitOfMeasure | unit_of_measure | Direct mapping |

