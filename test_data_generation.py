#!/usr/bin/env python3
"""
Test script for Azure cost data generation functionality.
This script tests the core data generation logic from the first notebook.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
import json
import os

def generate_azure_cost_data(num_records: int = 1000, start_date: str = '2023-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
    """
    Generate sample Azure cost management data with realistic patterns and trends.
    """
    
    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Azure regions - Focus on East US (90%) and South Central US (10%)
    AZURE_REGIONS = ['East US', 'South Central US']
    REGION_WEIGHTS = [0.9, 0.1]  # 90% East US, 10% South Central US
    
    # Meter categories and subcategories
    METER_CATEGORIES = {
        'Compute': ['Virtual Machines', 'Container Instances', 'App Service', 'Functions', 'Batch'],
        'Storage': ['Blob Storage', 'File Storage', 'Disk Storage', 'Archive Storage', 'Data Lake'],
        'Network': ['Bandwidth', 'Load Balancer', 'VPN Gateway', 'Application Gateway', 'CDN'],
        'Database': ['SQL Database', 'Cosmos DB', 'Redis Cache', 'PostgreSQL', 'MySQL'],
        'Analytics': ['Data Factory', 'Stream Analytics', 'HDInsight', 'Synapse', 'Power BI'],
        'AI/ML': ['Cognitive Services', 'Machine Learning', 'Bot Service', 'Computer Vision', 'Speech Services'],
        'Security': ['Key Vault', 'Security Center', 'Azure AD', 'Sentinel', 'Defender'],
        'Management': ['Monitor', 'Log Analytics', 'Backup', 'Site Recovery', 'Policy']
    }
    
    # Service tiers
    SERVICE_TIERS = ['Basic', 'Standard', 'Premium', 'Free', 'Consumption']
    
    # Resource types
    RESOURCE_TYPES = [
        'Microsoft.Compute/virtualMachines',
        'Microsoft.Storage/storageAccounts',
        'Microsoft.Network/loadBalancers',
        'Microsoft.Sql/servers',
        'Microsoft.Web/sites',
        'Microsoft.ContainerService/managedClusters',
        'Microsoft.CognitiveServices/accounts',
        'Microsoft.KeyVault/vaults'
    ]
    
    # Currency codes - USD only
    CURRENCIES = ['USD']
    
    # Units of measure
    UNITS_OF_MEASURE = [
        '1 Hour', '1 GB', '1 GB-Month', '1 GB-Hour', '1 TB', '1 TB-Month',
        '1 Request', '1 Transaction', '1 API Call', '1 Unit', '1 Node',
        '1 Instance', '1 Core', '1 vCPU', '1 GB-Second'
    ]
    
    # Generate data
    data = []
    
    for i in range(num_records):
        # Generate random date within range
        random_days = random.randint(0, (end_dt - start_dt).days)
        usage_date = start_dt + timedelta(days=random_days)
        
        # Add some randomness to the time
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        usage_datetime = usage_date.replace(hour=random_hours, minute=random_minutes)
        
        # Select random category and subcategory
        category = random.choice(list(METER_CATEGORIES.keys()))
        subcategory = random.choice(METER_CATEGORIES[category])
        
        # Generate realistic resource rates based on category
        base_rates = {
            'Compute': (0.05, 2.0),
            'Storage': (0.001, 0.1),
            'Network': (0.01, 0.5),
            'Database': (0.1, 5.0),
            'Analytics': (0.02, 1.0),
            'AI/ML': (0.01, 3.0),
            'Security': (0.05, 2.0),
            'Management': (0.01, 0.5)
        }
        
        min_rate, max_rate = base_rates[category]
        resource_rate = round(random.uniform(min_rate, max_rate), 4)
        
        # Generate usage quantity (higher for some categories)
        quantity_multipliers = {
            'Compute': (1, 1000),
            'Storage': (1, 10000),
            'Network': (1, 1000),
            'Database': (1, 100),
            'Analytics': (1, 1000),
            'AI/ML': (1, 10000),
            'Security': (1, 100),
            'Management': (1, 1000)
        }
        
        min_qty, max_qty = quantity_multipliers[category]
        usage_quantity = round(random.uniform(min_qty, max_qty), 2)
        
        # Calculate pre-tax cost
        pre_tax_cost = round(usage_quantity * resource_rate, 4)
        
        # Generate seasonal patterns (higher costs in certain months)
        month = usage_datetime.month
        seasonal_multiplier = 1.0
        if month in [11, 12, 1]:  # Holiday season
            seasonal_multiplier = 1.3
        elif month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.1
        
        pre_tax_cost *= seasonal_multiplier
        
        # Generate weekend/weekday patterns
        if usage_datetime.weekday() >= 5:  # Weekend
            pre_tax_cost *= 0.7
        
        # Select region based on weights (90% East US, 10% South Central US)
        resource_location = np.random.choice(AZURE_REGIONS, p=REGION_WEIGHTS)
        
        # Generate resource group names
        resource_groups = [
            f'rg-{category.lower()}-{random.randint(1, 10)}',
            f'rg-prod-{random.randint(1, 5)}',
            f'rg-dev-{random.randint(1, 3)}',
            f'rg-test-{random.randint(1, 2)}',
            f'rg-shared-{random.randint(1, 3)}'
        ]
        
        # Generate tags
        tags = {
            'Environment': random.choice(['Production', 'Development', 'Test', 'Staging']),
            'Owner': f'team-{random.choice(["backend", "frontend", "data", "devops"])}',
            'Project': f'project-{random.randint(1, 10)}',
            'CostCenter': f'CC-{random.randint(100, 999)}'
        }
        
        record = {
            'SubscriptionGuid': str(uuid.uuid4()),
            'ResourceGroup': random.choice(resource_groups),
            'ResourceLocation': resource_location,
            'UsageDateTime': usage_datetime,
            'MeterCategory': category,
            'MeterSubCategory': subcategory,
            'MeterId': f'meter-{category.lower()}-{random.randint(1000, 9999)}',
            'MeterName': f'{subcategory} - {resource_location}',
            'MeterRegion': resource_location,
            'UsageQuantity': usage_quantity,
            'ResourceRate': resource_rate,
            'PreTaxCost': round(pre_tax_cost, 4),
            'ConsumedService': f'Microsoft.{category}',
            'ResourceType': random.choice(RESOURCE_TYPES),
            'InstanceId': f'instance-{random.randint(10000, 99999)}',
            'Tags': json.dumps(tags),
            'OfferId': f'MS-AZR-{random.randint(1000, 9999)}',
            'AdditionalInfo': f'Additional info for {category}',
            'ServiceInfo1': f'Service info 1 - {random.randint(1, 100)}',
            'ServiceInfo2': f'Service info 2 - {random.randint(1, 100)}',
            'ServiceName': f'Azure {subcategory}',
            'ServiceTier': random.choice(SERVICE_TIERS),
            'Currency': 'USD',  # USD only
            'UnitOfMeasure': random.choice(UNITS_OF_MEASURE)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Test the data generation functionality."""
    print("🚀 Testing Azure Cost Data Generation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate test data
    print("Generating test data...")
    df = generate_azure_cost_data(num_records=1000, start_date='2023-01-01', end_date='2023-12-31')
    
    print(f"✅ Generated {len(df)} records successfully!")
    print(f"Date range: {df['UsageDateTime'].min()} to {df['UsageDateTime'].max()}")
    print(f"Total cost: ${df['PreTaxCost'].sum():,.2f}")
    
    # Display basic statistics
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display sample data
    print("\n=== Sample Data ===")
    print(df.head())
    
    # Display cost analysis by category
    print("\n=== Cost Analysis by Category ===")
    category_analysis = df.groupby('MeterCategory').agg({
        'PreTaxCost': ['sum', 'mean', 'count'],
        'UsageQuantity': ['sum', 'mean']
    }).round(4)
    
    category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns]
    category_analysis = category_analysis.sort_values('PreTaxCost_sum', ascending=False)
    print(category_analysis)
    
    # Create data directory and save sample
    os.makedirs('data', exist_ok=True)
    sample_path = 'data/test_sample_azure_costs.csv'
    df.to_csv(sample_path, index=False)
    print(f"\n✅ Sample data saved to: {sample_path}")
    
    print("\n🎉 Data generation test completed successfully!")
    print("📊 Ready to run the full notebooks!")

if __name__ == "__main__":
    main()
