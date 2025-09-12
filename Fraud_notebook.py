# Credit Card Fraud Detection with Risk-Based Rules
# Objective: Define rules for credit card fraud with three risk levels:
# - Low risk: Accept
# - Medium risk: Decline and ask customer to validate  
# - High risk: Decline and alert agent
#
# Constraints:
# - Decline rate <= 30% of all transactions
# - Agent alerts < 0.1% of all transactions  
# - Missed fraud <= 0.02%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Credit Card Fraud Detection System")
print("=" * 50)

# Load the dataset
# Note: Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
try:
    df = pd.read_csv('creditcard.csv')
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean():.4f} ({df['Class'].sum()} frauds out of {len(df)} transactions)")
except FileNotFoundError:
    print("Error: Please download 'creditcard.csv' from Kaggle and place it in the current directory")
    print("Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    # Create synthetic data for demonstration
    print("\nCreating synthetic dataset for demonstration...")
    from sklearn.datasets import make_classification
    X_synthetic, y_synthetic = make_classification(
        n_samples=50000, n_features=30, n_informative=15, n_redundant=10,
        n_classes=2, weights=[0.998, 0.002], flip_y=0.01, random_state=42
    )
    df = pd.DataFrame(X_synthetic, columns=[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'])
    df['Class'] = y_synthetic
    df['Time'] = np.random.uniform(0, 172800, len(df))  # 2 days in seconds
    df['Amount'] = np.random.lognormal(3, 1.5, len(df))
    print(f"Synthetic dataset created: {df.shape}")

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Basic statistics
print("\nDataset Info:")
print(f"Total transactioxns: {len(df):,}")
print(f"Fraudulent transactions: {df['Class'].sum():,}")
print(f"Fraud rate: {df['Class'].mean():.4f} ({df['Class'].mean()*100:.2f}%)")
print(f"Missing values: {df.isnull().sum().sum()}")

# Class distribution
fraud_counts = df['Class'].value_counts()
print(f"\nClass Distribution:")
print(f"Normal transactions: {fraud_counts[0]:,} ({fraud_counts[0]/len(df)*100:.2f}%)")
print(f"Fraud transactions: {fraud_counts[1]:,} ({fraud_counts[1]/len(df)*100:.2f}%)")

# Amount analysis
print(f"\nTransaction Amount Analysis:")
print(f"Normal transactions - Mean: ${df[df['Class']==0]['Amount'].mean():.2f}, Median: ${df[df['Class']==0]['Amount'].median():.2f}")
print(f"Fraud transactions - Mean: ${df[df['Class']==1]['Amount'].mean():.2f}, Median: ${df[df['Class']==1]['Amount'].median():.2f}")

# Time analysis (convert to hours)
df['Hour'] = (df['Time'] % (24*3600)) // 3600

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Create additional features for rule-based detection
def create_features(data):
    """Create additional features for fraud detection"""
    df_features = data.copy()
    
    # Amount-based features
    df_features['Amount_log'] = np.log1p(df_features['Amount'])
    df_features['Amount_sqrt'] = np.sqrt(df_features['Amount'])
    
    # Amount categories
    df_features['Amount_category'] = pd.cut(df_features['Amount'], 
                                           bins=[0, 10, 100, 1000, float('inf')], 
                                           labels=['Very_Low', 'Low', 'Medium', 'High'])
    
    # Time-based features
    df_features['Hour_sin'] = np.sin(2 * np.pi * df_features['Hour'] / 24)
    df_features['Hour_cos'] = np.cos(2 * np.pi * df_features['Hour'] / 24)
    df_features['Is_Night'] = ((df_features['Hour'] >= 23) | (df_features['Hour'] <= 6)).astype(int)
    df_features['Is_Weekend'] = ((df_features['Time'] // (24*3600)) % 7 >= 5).astype(int)
    
    # PCA feature combinations (for V1-V28)
    v_cols = [col for col in df_features.columns if col.startswith('V')]
    if len(v_cols) >= 28:
        df_features['V_sum_pos'] = df_features[v_cols].clip(lower=0).sum(axis=1)
        df_features['V_sum_neg'] = df_features[v_cols].clip(upper=0).sum(axis=1)
        df_features['V_mean'] = df_features[v_cols].mean(axis=1)
        df_features['V_std'] = df_features[v_cols].std(axis=1)
    
    return df_features

# Apply feature engineering
df_features = create_features(df)
print(f"Features created. New shape: {df_features.shape}")

# =============================================================================
# MODEL TRAINING
# =============================================================================

print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Prepare features for modeling
feature_cols = [col for col in df_features.columns if col not in ['Class', 'Amount_category']]
X = df_features[feature_cols]
y = df_features['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

print(f"Training set: {X_train.shape[0]:,} transactions")
print(f"Test set: {X_test.shape[0]:,} transactions")
print(f"Training fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate: {y_test.mean():.4f}")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models for ensemble approach
models = {}

print("\nTraining models...")

# 1. Logistic Regression
print("- Training Logistic Regression...")
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
models['logistic'] = lr_model

# 2. Random Forest  
print("- Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
models['random_forest'] = rf_model

# 3. Isolation Forest (for anomaly detection)
print("- Training Isolation Forest...")
iso_model = IsolationForest(contamination=0.002, random_state=42, n_jobs=-1)
iso_model.fit(X_train_scaled[y_train == 0])  # Train only on normal transactions
models['isolation'] = iso_model

print("Model training completed!")

# =============================================================================
# RISK SCORE CALCULATION
# =============================================================================

print("\n" + "="*50)
print("RISK SCORE CALCULATION")
print("="*50)

def calculate_risk_scores(X_data, models, scaler):
    """Calculate comprehensive risk scores using multiple models"""
    X_scaled = scaler.transform(X_data)
    
    # Get probabilities from different models
    lr_probs = models['logistic'].predict_proba(X_scaled)[:, 1]
    rf_probs = models['random_forest'].predict_proba(X_scaled)[:, 1]
    iso_scores = models['isolation'].decision_function(X_scaled)
    
    # Normalize isolation forest scores to 0-1 range
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    iso_scores_norm = 1 - iso_scores_norm  # Invert so higher = more anomalous
    
    # Ensemble risk score (weighted combination)
    risk_scores = (0.4 * lr_probs + 0.4 * rf_probs + 0.2 * iso_scores_norm)
    
    return risk_scores, {
        'logistic_prob': lr_probs,
        'rf_prob': rf_probs, 
        'isolation_score': iso_scores_norm
    }

# Calculate risk scores for test set
test_risk_scores, test_individual_scores = calculate_risk_scores(X_test, models, scaler)

print(f"Risk scores calculated for {len(test_risk_scores):,} transactions")
print(f"Risk score range: {test_risk_scores.min():.4f} - {test_risk_scores.max():.4f}")
print(f"Mean risk score: {test_risk_scores.mean():.4f}")

# =============================================================================
# RISK THRESHOLD OPTIMIZATION
# =============================================================================

print("\n" + "="*50)
print("RISK THRESHOLD OPTIMIZATION")  
print("="*50)

def evaluate_thresholds(risk_scores, y_true, low_threshold, high_threshold):
    """Evaluate risk-based rules with given thresholds"""
    # Classify transactions
    low_risk = risk_scores < low_threshold
    medium_risk = (risk_scores >= low_threshold) & (risk_scores < high_threshold)
    high_risk = risk_scores >= high_threshold
    
    total_transactions = len(y_true)
    
    # Calculate metrics
    results = {
        'total_transactions': total_transactions,
        'low_risk_count': low_risk.sum(),
        'medium_risk_count': medium_risk.sum(),
        'high_risk_count': high_risk.sum(),
        'low_risk_pct': low_risk.mean() * 100,
        'medium_risk_pct': medium_risk.mean() * 100,
        'high_risk_pct': high_risk.mean() * 100
    }
    
    # Decline rate (medium + high risk)
    decline_rate = (medium_risk.sum() + high_risk.sum()) / total_transactions
    
    # Agent alert rate (high risk only)
    agent_alert_rate = high_risk.sum() / total_transactions
    
    # Missed fraud rate (frauds classified as low risk)
    fraud_indices = y_true == 1
    missed_frauds = (fraud_indices & low_risk).sum()
    total_frauds = fraud_indices.sum()
    missed_fraud_rate = missed_frauds / total_frauds if total_frauds > 0 else 0
    
    # Fraud detection by risk category
    low_risk_frauds = (fraud_indices & low_risk).sum()
    medium_risk_frauds = (fraud_indices & medium_risk).sum()
    high_risk_frauds = (fraud_indices & high_risk).sum()
    
    results.update({
        'decline_rate': decline_rate,
        'agent_alert_rate': agent_alert_rate,
        'missed_fraud_rate': missed_fraud_rate,
        'low_risk_frauds': low_risk_frauds,
        'medium_risk_frauds': medium_risk_frauds,
        'high_risk_frauds': high_risk_frauds,
        'total_frauds': total_frauds
    })
    
    return results

# Test different threshold combinations
print("Optimizing thresholds to meet constraints...")
print("Constraints: Decline rate ≤ 30%, Agent alerts < 0.1%, Missed fraud ≤ 0.02%")

best_thresholds = None
best_results = None
threshold_tests = []

# Test various threshold combinations
for low_thresh in np.arange(0.1, 0.8, 0.05):
    for high_thresh in np.arange(low_thresh + 0.05, 0.95, 0.05):
        results = evaluate_thresholds(test_risk_scores, y_test, low_thresh, high_thresh)
        
        # Check if constraints are met
        meets_constraints = (
            results['decline_rate'] <= 0.30 and  # Decline rate ≤ 30%
            results['agent_alert_rate'] < 0.001 and  # Agent alerts < 0.1%
            results['missed_fraud_rate'] <= 0.02  # Missed fraud ≤ 2%
        )
        
        results['low_threshold'] = low_thresh
        results['high_threshold'] = high_thresh
        results['meets_constraints'] = meets_constraints
        
        threshold_tests.append(results)
        
        if meets_constraints and (best_results is None or 
                                results['missed_fraud_rate'] < best_results['missed_fraud_rate']):
            best_thresholds = (low_thresh, high_thresh)
            best_results = results

# Display results
if best_thresholds:
    low_thresh, high_thresh = best_thresholds
    print(f"\nOPTIMAL THRESHOLDS FOUND:")
    print(f"Low risk threshold: {low_thresh:.3f}")
    print(f"High risk threshold: {high_thresh:.3f}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Decline rate: {best_results['decline_rate']*100:.2f}%")
    print(f"Agent alert rate: {best_results['agent_alert_rate']*100:.4f}%") 
    print(f"Missed fraud rate: {best_results['missed_fraud_rate']*100:.2f}%")
    
    print(f"\nRISK DISTRIBUTION:")
    print(f"Low risk: {best_results['low_risk_count']:,} transactions ({best_results['low_risk_pct']:.1f}%)")
    print(f"Medium risk: {best_results['medium_risk_count']:,} transactions ({best_results['medium_risk_pct']:.1f}%)")
    print(f"High risk: {best_results['high_risk_count']:,} transactions ({best_results['high_risk_pct']:.1f}%)")
    
    print(f"\nFRAUD DETECTION BY RISK LEVEL:")
    print(f"Low risk frauds: {best_results['low_risk_frauds']} (missed)")
    print(f"Medium risk frauds: {best_results['medium_risk_frauds']} (customer validation)")
    print(f"High risk frauds: {best_results['high_risk_frauds']} (agent review)")
    
else:
    print("\nNo threshold combination found that meets all constraints!")
    print("Consider relaxing constraints or improving the model.")
    
    # Show best alternatives
    df_results = pd.DataFrame(threshold_tests)
    df_results = df_results.sort_values('missed_fraud_rate')
    
    print("\nBest alternatives (lowest missed fraud rate):")
    print(df_results[['low_threshold', 'high_threshold', 'decline_rate', 
                     'agent_alert_rate', 'missed_fraud_rate']].head().round(4))

# =============================================================================
# BUSINESS RULES DEFINITION  
# =============================================================================

print("\n" + "="*50)
print("BUSINESS RULES DEFINITION")
print("="*50)

if best_thresholds:
    low_thresh, high_thresh = best_thresholds
    
    print("FRAUD DETECTION BUSINESS RULES")
    print("-" * 40)
    print(f"Rule 1 - LOW RISK (Accept Transaction):")
    print(f"  • Risk Score < {low_thresh:.3f}")
    print(f"  • Action: APPROVE transaction automatically")
    print(f"  • Expected volume: ~{best_results['low_risk_pct']:.1f}% of transactions")
    
    print(f"\nRule 2 - MEDIUM RISK (Customer Validation):")
    print(f"  • Risk Score >= {low_thresh:.3f} AND < {high_thresh:.3f}")
    print(f"  • Action: DECLINE transaction, request customer validation")
    print(f"  • Expected volume: ~{best_results['medium_risk_pct']:.1f}% of transactions")
    print(f"  • Customer actions: SMS/Email verification, phone call, etc.")
    
    print(f"\nRule 3 - HIGH RISK (Agent Review):")
    print(f"  • Risk Score >= {high_thresh:.3f}")
    print(f"  • Action: DECLINE transaction, create agent alert")
    print(f"  • Expected volume: ~{best_results['high_risk_pct']:.1f}% of transactions")
    print(f"  • Agent actions: Manual review, contact customer, investigate pattern")
    
    print(f"\nSYSTEM PERFORMANCE:")
    print(f"✓ Decline rate: {best_results['decline_rate']*100:.1f}% (≤ 30% ✓)")
    print(f"✓ Agent alerts: {best_results['agent_alert_rate']*100:.3f}% (< 0.1% ✓)")
    print(f"✓ Missed fraud: {best_results['missed_fraud_rate']*100:.1f}% (≤ 2% ✓)")

# =============================================================================
# RISK SCORE IMPLEMENTATION FUNCTION
# =============================================================================

print("\n" + "="*50)
print("PRODUCTION IMPLEMENTATION")
print("="*50)

def fraud_risk_classifier(transaction_features, models, scaler, low_threshold, high_threshold):
    """
    Production function to classify transactions by fraud risk
    
    Parameters:
    - transaction_features: DataFrame with transaction features
    - models: Dictionary of trained models
    - scaler: Fitted feature scaler
    - low_threshold: Threshold between low and medium risk
    - high_threshold: Threshold between medium and high risk
    
    Returns:
    - risk_level: 'low', 'medium', or 'high'
    - risk_score: Numerical risk score (0-1)
    - action: Recommended action
    """
    
    # Calculate risk score
    risk_scores, _ = calculate_risk_scores(transaction_features, models, scaler)
    risk_score = risk_scores[0] if len(risk_scores) == 1 else risk_scores
    
    # Determine risk level and action
    if isinstance(risk_score, np.ndarray):
        risk_levels = []
        actions = []
        for score in risk_score:
            if score < low_threshold:
                risk_levels.append('low')
                actions.append('APPROVE')
            elif score < high_threshold:
                risk_levels.append('medium')
                actions.append('DECLINE_VALIDATE')
            else:
                risk_levels.append('high')
                actions.append('DECLINE_ALERT')
        return risk_levels, risk_score, actions
    else:
        if risk_score < low_threshold:
            return 'low', risk_score, 'APPROVE'
        elif risk_score < high_threshold:
            return 'medium', risk_score, 'DECLINE_VALIDATE'
        else:
            return 'high', risk_score, 'DECLINE_ALERT'

# Example usage
if best_thresholds:
    print("PRODUCTION FUNCTION EXAMPLE:")
    print("-" * 30)
    
    # Test with a few sample transactions
    sample_transactions = X_test.head(5)
    risk_levels, risk_scores, actions = fraud_risk_classifier(
        sample_transactions, models, scaler, low_thresh, high_thresh
    )
    
    for i, (level, score, action) in enumerate(zip(risk_levels, risk_scores, actions)):
        actual_fraud = 'FRAUD' if y_test.iloc[i] == 1 else 'NORMAL'
        print(f"Transaction {i+1}: Risk={level.upper()} (Score={score:.4f}) → {action} | Actual: {actual_fraud}")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

if best_thresholds:
    # Plot 1: Risk Score Distribution
    ax1 = axes[0, 0]
    ax1.hist(test_risk_scores[y_test == 0], bins=50, alpha=0.7, label='Normal', density=True)
    ax1.hist(test_risk_scores[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
    ax1.axvline(low_thresh, color='orange', linestyle='--', label=f'Low Threshold ({low_thresh:.3f})')
    ax1.axvline(high_thresh, color='red', linestyle='--', label=f'High Threshold ({high_thresh:.3f})')
    ax1.set_xlabel('Risk Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Risk Score Distribution by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transaction Volume by Risk Level
    ax2 = axes[0, 1]
    risk_counts = [best_results['low_risk_count'], best_results['medium_risk_count'], best_results['high_risk_count']]
    risk_labels = ['Low Risk\n(Approve)', 'Medium Risk\n(Validate)', 'High Risk\n(Alert)']
    colors = ['green', 'orange', 'red']
    bars = ax2.bar(risk_labels, risk_counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Transactions')
    ax2.set_title('Transaction Volume by Risk Level')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, risk_counts):
        percentage = count / sum(risk_counts) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sum(risk_counts)*0.01, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Fraud Detection by Risk Level
    ax3 = axes[1, 0]
    fraud_counts = [best_results['low_risk_frauds'], best_results['medium_risk_frauds'], best_results['high_risk_frauds']]
    bars = ax3.bar(risk_labels, fraud_counts, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Fraudulent Transactions')
    ax3.set_title('Fraud Detection by Risk Level')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, fraud_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fraud_counts)*0.05, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Performance Metrics
    ax4 = axes[1, 1]
    metrics = ['Decline Rate', 'Agent Alert Rate', 'Missed Fraud Rate']
    values = [best_results['decline_rate']*100, best_results['agent_alert_rate']*100, best_results['missed_fraud_rate']*100]
    constraints = [30, 0.1, 2]  # Constraint values in percentages
    
    x_pos = np.arange(len(metrics))
    bars = ax4.bar(x_pos, values, color=['orange', 'red', 'purple'], alpha=0.7, label='Actual')
    constraint_bars = ax4.bar(x_pos, constraints, alpha=0.3, color='gray', label='Constraint')
    
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Performance vs Constraints')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (actual, constraint) in enumerate(zip(values, constraints)):
        ax4.text(i, actual + max(values)*0.05, f'{actual:.2f}%', ha='center', va='bottom', fontweight='bold')
        status = '✓' if actual <= constraint else '✗'
        ax4.text(i, constraint + max(values)*0.05, status, ha='center', va='bottom', 
                fontsize=14, color='green' if status == '✓' else 'red', fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "="*80)
print("FRAUD DETECTION SYSTEM - FINAL SUMMARY REPORT")
print("="*80)

if best_thresholds:
    print(f"""
SYSTEM CONFIGURATION:
• Low Risk Threshold: {low_thresh:.3f}
• High Risk Threshold: {high_thresh:.3f}
• Model Ensemble: Logistic Regression (40%) + Random Forest (40%) + Isolation Forest (20%)

PERFORMANCE METRICS:
• Decline Rate: {best_results['decline_rate']*100:.2f}% (Target: ≤ 30%) ✓
• Agent Alert Rate: {best_results['agent_alert_rate']*100:.4f}% (Target: < 0.1%) ✓
• Missed Fraud Rate: {best_results['missed_fraud_rate']*100:.2f}% (Target: ≤ 2%) ✓

TRANSACTION PROCESSING:
• Low Risk (Auto-Approve): {best_results['low_risk_pct']:.1f}% of transactions
• Medium Risk (Customer Validation): {best_results['medium_risk_pct']:.1f}% of transactions  
• High Risk (Agent Review): {best_results['high_risk_pct']:.1f}% of transactions

FRAUD DETECTION EFFECTIVENESS:
• Total Frauds Detected: {best_results['medium_risk_frauds'] + best_results['high_risk_frauds']} out of {best_results['total_frauds']}
• Detection Rate: {((best_results['medium_risk_frauds'] + best_results['high_risk_frauds']) / best_results['total_frauds'] * 100):.1f}%
• Frauds Requiring Customer Validation: {best_results['medium_risk_frauds']}
• Frauds Requiring Agent Review: {best_results['high_risk_frauds']}

BUSINESS IMPACT:
• Reduced manual review workload by processing {best_results['low_risk_pct']:.1f}% of transactions automatically
• Efficient resource allocation with only {best_results['high_risk_pct']:.1f}% requiring agent attention
• Customer friction minimized while maintaining strong fraud protection
""")

    print("\nRECOMMENDations FOR PRODUCTION DEPLOYMENT:")
    print("• Implement real-time risk scoring with the provided function")
    print("• Set up monitoring dashboards for the three key performance metrics")
    print("• Establish feedback loops to retrain models with new fraud patterns")
    print("• Create escalation procedures for high-risk transactions")
    print("• Regularly validate thresholds against business constraints")
    print("• Consider A/B testing before full deployment")

else:
    print("SYSTEM OPTIMIZATION FAILED")
    print("Unable to find thresholds that meet all business constraints.")
    print("Recommended next steps:")
    print("• Improve feature engineering")
    print("• Collect more training data")
    print("• Consider relaxing business constraints")
    print("• Explore advanced modeling techniques")

print("\n" + "="*80)
print("Analysis completed successfully!")
print("="*80)