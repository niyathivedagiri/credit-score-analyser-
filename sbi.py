import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math

# Configure page
st.set_page_config(
    page_title="Credit Score Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .score-excellent { color: #27ae60; font-weight: bold; }
    .score-good { color: #3498db; font-weight: bold; }
    .score-fair { color: #f39c12; font-weight: bold; }
    .score-poor { color: #e74c3c; font-weight: bold; }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .improvement-tip {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin: 0.5rem 0;
    }
    
    .warning-tip {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 8px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class CreditScoreAnalyzer:
    def __init__(self):
        self.weights = {
            'payment_history': 0.35,
            'credit_utilization': 0.30,
            'credit_history_length': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }
    
    def calculate_score(self, user_data):
        """Advanced credit score calculation with AI-like logic"""
        base_score = 300
        max_possible = 600  # 900 - 300
        
        # Payment History Score (35%)
        payment_scores = {
            'Always on time': 100,
            'Mostly on time (1-2 late payments/year)': 85,
            'Sometimes late (3-5 late payments/year)': 60,
            'Frequently late (6+ late payments/year)': 30,
            'Defaults/Charge-offs': 0
        }
        payment_score = payment_scores.get(user_data['payment_history'], 50)
        
        # Credit Utilization Score (30%)
        utilization = user_data['credit_utilization']
        if utilization <= 10:
            util_score = 100
        elif utilization <= 30:
            util_score = 85
        elif utilization <= 50:
            util_score = 60
        elif utilization <= 70:
            util_score = 40
        else:
            util_score = 20
        
        # Credit History Length Score (15%)
        history_years = user_data['credit_history_years']
        if history_years >= 15:
            history_score = 100
        elif history_years >= 10:
            history_score = 85
        elif history_years >= 5:
            history_score = 70
        elif history_years >= 2:
            history_score = 50
        else:
            history_score = 30
        
        # Credit Mix Score (10%)
        credit_types = user_data['credit_types']
        mix_score = min(100, credit_types * 25)
        
        # New Credit Score (10%)
        recent_inquiries = user_data['recent_inquiries']
        if recent_inquiries == 0:
            new_credit_score = 100
        elif recent_inquiries <= 2:
            new_credit_score = 80
        elif recent_inquiries <= 4:
            new_credit_score = 60
        else:
            new_credit_score = 30
        
        # Calculate weighted score
        weighted_score = (
            payment_score * self.weights['payment_history'] +
            util_score * self.weights['credit_utilization'] +
            history_score * self.weights['credit_history_length'] +
            mix_score * self.weights['credit_mix'] +
            new_credit_score * self.weights['new_credit']
        )
        
        # Add income and employment stability factors
        income_factor = min(50, user_data['annual_income'] / 10000)
        employment_factors = {
            'Salaried (Private)': 40,
            'Salaried (Government)': 50,
            'Self-employed': 30,
            'Business owner': 35,
            'Retired': 25,
            'Student': 15
        }
        employment_factor = employment_factors.get(user_data['employment_type'], 25)
        
        # Final score calculation
        final_score = base_score + (weighted_score / 100 * max_possible)
        final_score += income_factor + employment_factor
        
        # Add some realistic variance
        variance = np.random.normal(0, 15)
        final_score += variance
        
        # Ensure score is within valid range
        final_score = max(300, min(900, int(final_score)))
        
        return final_score, {
            'payment_history': payment_score,
            'credit_utilization': util_score,
            'credit_history_length': history_score,
            'credit_mix': mix_score,
            'new_credit': new_credit_score,
            'income_factor': income_factor,
            'employment_factor': employment_factor
        }
    
    def get_score_category(self, score):
        if score >= 750:
            return "Excellent", "#27ae60"
        elif score >= 650:
            return "Good", "#3498db"
        elif score >= 550:
            return "Fair", "#f39c12"
        else:
            return "Poor", "#e74c3c"
    
    def generate_insights(self, score, user_data, component_scores):
        """AI-powered insights generation"""
        category, _ = self.get_score_category(score)
        insights = []
        
        # Payment history insights
        if component_scores['payment_history'] < 70:
            insights.append({
                'type': 'warning',
                'title': 'Payment History Needs Attention',
                'message': 'Your payment history is the most important factor. Set up auto-pay to ensure on-time payments.',
                'impact': 'High'
            })
        
        # Credit utilization insights
        if user_data['credit_utilization'] > 30:
            insights.append({
                'type': 'warning',
                'title': 'High Credit Utilization',
                'message': f'Your {user_data["credit_utilization"]}% utilization is high. Aim for under 30% for better scores.',
                'impact': 'High'
            })
        
        # Credit history insights
        if user_data['credit_history_years'] < 5:
            insights.append({
                'type': 'tip',
                'title': 'Build Credit History',
                'message': 'Keep older accounts open and consider becoming an authorized user on family accounts.',
                'impact': 'Medium'
            })
        
        # Credit mix insights
        if user_data['credit_types'] < 3:
            insights.append({
                'type': 'tip',
                'title': 'Diversify Credit Types',
                'message': 'Having different types of credit (cards, loans, etc.) can improve your score.',
                'impact': 'Low'
            })
        
        return insights

class LoanPredictor:
    def __init__(self):
        self.loan_types = {
            'Personal Loan': {
                'base_rate': 12.0,
                'max_amount_multiplier': 20,  # 20x monthly income
                'min_score': 650,
                'max_tenure': 5
            },
            'Home Loan': {
                'base_rate': 8.5,
                'max_amount_multiplier': 80,  # 80x monthly income
                'min_score': 700,
                'max_tenure': 30
            },
            'Car Loan': {
                'base_rate': 9.5,
                'max_amount_multiplier': 40,  # 40x monthly income
                'min_score': 650,
                'max_tenure': 7
            },
            'Education Loan': {
                'base_rate': 10.5,
                'max_amount_multiplier': 60,  # 60x monthly income
                'min_score': 600,
                'max_tenure': 15
            },
            'Business Loan': {
                'base_rate': 13.0,
                'max_amount_multiplier': 30,  # 30x monthly income
                'min_score': 700,
                'max_tenure': 5
            },
            'Gold Loan': {
                'base_rate': 11.0,
                'max_amount_multiplier': 15,  # 15x monthly income
                'min_score': 500,
                'max_tenure': 3
            },
            'Credit Card': {
                'base_rate': 42.0,  # Annual rate
                'max_amount_multiplier': 25,  # 25x monthly income
                'min_score': 650,
                'max_tenure': 0  # Revolving
            }
        }
        
        self.bank_preferences = {
            'SBI': {'rate_adjustment': 0.0, 'amount_multiplier': 1.0},
            'HDFC': {'rate_adjustment': 0.5, 'amount_multiplier': 1.2},
            'ICICI': {'rate_adjustment': 0.3, 'amount_multiplier': 1.1},
            'Axis Bank': {'rate_adjustment': 0.7, 'amount_multiplier': 1.15},
            'Kotak Mahindra': {'rate_adjustment': 0.8, 'amount_multiplier': 1.1},
            'IDFC First': {'rate_adjustment': 1.0, 'amount_multiplier': 0.9},
            'Yes Bank': {'rate_adjustment': 1.2, 'amount_multiplier': 0.95},
            'IndusInd Bank': {'rate_adjustment': 0.9, 'amount_multiplier': 1.05}
        }
    
    def calculate_interest_rate(self, loan_type, credit_score, employment_type, existing_loans):
        """Calculate personalized interest rate based on risk factors"""
        base_rate = self.loan_types[loan_type]['base_rate']
        
        # Credit score impact
        if credit_score >= 800:
            score_adjustment = -2.0
        elif credit_score >= 750:
            score_adjustment = -1.0
        elif credit_score >= 700:
            score_adjustment = 0.0
        elif credit_score >= 650:
            score_adjustment = 1.5
        elif credit_score >= 600:
            score_adjustment = 3.0
        else:
            score_adjustment = 5.0
        
        # Employment type impact
        employment_adjustments = {
            'Salaried (Government)': -1.0,
            'Salaried (Private)': 0.0,
            'Self-employed': 1.5,
            'Business owner': 2.0,
            'Retired': 1.0,
            'Student': 2.5
        }
        employment_adjustment = employment_adjustments.get(employment_type, 1.0)
        
        # Existing loans impact
        existing_loan_adjustment = existing_loans * 0.5
        
        final_rate = base_rate + score_adjustment + employment_adjustment + existing_loan_adjustment
        return max(6.0, min(48.0, final_rate))  # Cap between 6% and 48%
    
    def calculate_loan_amount(self, loan_type, monthly_income, credit_score, existing_loans):
        """Calculate maximum eligible loan amount"""
        multiplier = self.loan_types[loan_type]['max_amount_multiplier']
        
        # Credit score impact on loan amount
        if credit_score >= 800:
            score_multiplier = 1.3
        elif credit_score >= 750:
            score_multiplier = 1.2
        elif credit_score >= 700:
            score_multiplier = 1.0
        elif credit_score >= 650:
            score_multiplier = 0.8
        elif credit_score >= 600:
            score_multiplier = 0.6
        else:
            score_multiplier = 0.4
        
        # Reduce amount based on existing loans
        existing_loan_reduction = 1 - (existing_loans * 0.1)
        existing_loan_reduction = max(0.3, existing_loan_reduction)
        
        max_amount = monthly_income * multiplier * score_multiplier * existing_loan_reduction
        
        # Set realistic caps for different loan types
        caps = {
            'Personal Loan': 5000000,  # 50 Lakhs
            'Home Loan': 100000000,   # 10 Crores
            'Car Loan': 5000000,      # 50 Lakhs
            'Education Loan': 10000000, # 1 Crore
            'Business Loan': 50000000,  # 5 Crores
            'Gold Loan': 2000000,     # 20 Lakhs
            'Credit Card': 1000000    # 10 Lakhs
        }
        
        return min(max_amount, caps.get(loan_type, 5000000))
    
    def get_eligible_loans(self, user_data, credit_score):
        """Get all eligible loans with details"""
        monthly_income = user_data['annual_income'] / 12
        eligible_loans = []
        
        for loan_type, criteria in self.loan_types.items():
            if credit_score >= criteria['min_score']:
                interest_rate = self.calculate_interest_rate(
                    loan_type, credit_score, user_data['employment_type'], user_data['existing_loans']
                )
                
                max_amount = self.calculate_loan_amount(
                    loan_type, monthly_income, credit_score, user_data['existing_loans']
                )
                
                # Calculate EMI for different tenures
                tenures = []
                if criteria['max_tenure'] > 0:
                    tenure_options = [1, 2, 3, 5] if criteria['max_tenure'] >= 5 else [1, 2, 3]
                    if criteria['max_tenure'] >= 10:
                        tenure_options.extend([10, 15])
                    if criteria['max_tenure'] >= 20:
                        tenure_options.extend([20, 25, 30])
                    
                    for tenure in tenure_options:
                        if tenure <= criteria['max_tenure']:
                            monthly_rate = interest_rate / (12 * 100)
                            total_months = tenure * 12
                            
                            if monthly_rate > 0:
                                emi = (max_amount * monthly_rate * (1 + monthly_rate)**total_months) / \
                                      ((1 + monthly_rate)**total_months - 1)
                            else:
                                emi = max_amount / total_months
                            
                            tenures.append({
                                'tenure': tenure,
                                'emi': emi,
                                'total_amount': emi * total_months,
                                'total_interest': (emi * total_months) - max_amount
                            })
                
                # Get bank offers
                bank_offers = []
                for bank, preferences in self.bank_preferences.items():
                    bank_rate = interest_rate + preferences['rate_adjustment']
                    bank_amount = max_amount * preferences['amount_multiplier']
                    
                    bank_offers.append({
                        'bank': bank,
                        'rate': round(bank_rate, 2),
                        'max_amount': bank_amount,
                        'processing_fee': round(bank_amount * 0.02, 0)  # 2% processing fee
                    })
                
                # Sort by interest rate
                bank_offers.sort(key=lambda x: x['rate'])
                
                eligible_loans.append({
                    'loan_type': loan_type,
                    'interest_rate': round(interest_rate, 2),
                    'max_amount': round(max_amount, 0),
                    'max_tenure': criteria['max_tenure'],
                    'tenures': tenures,
                    'bank_offers': bank_offers[:5],  # Top 5 banks
                    'approval_probability': self.calculate_approval_probability(credit_score, criteria['min_score'])
                })
        
        return sorted(eligible_loans, key=lambda x: x['approval_probability'], reverse=True)
    
    def calculate_approval_probability(self, credit_score, min_score):
        """Calculate loan approval probability"""
        if credit_score < min_score:
            return 0
        
        score_excess = credit_score - min_score
        max_excess = 900 - min_score
        
        probability = 60 + (score_excess / max_excess) * 40  # 60-100% range
        return min(99, round(probability, 0))

def create_score_gauge(score):
    """Create a beautiful gauge chart for credit score"""
    analyzer = CreditScoreAnalyzer()
    category, color = analyzer.get_score_category(score)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Credit Score: {category}"},
        delta = {'reference': 650},
        gauge = {
            'axis': {'range': [None, 900]},
            'bar': {'color': color},
            'steps': [
                {'range': [300, 550], 'color': "lightgray"},
                {'range': [550, 650], 'color': "yellow"},
                {'range': [650, 750], 'color': "lightblue"},
                {'range': [750, 900], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 750
            }
        }
    ))
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_factor_breakdown(component_scores, weights):
    """Create factor breakdown visualization"""
    factors = list(component_scores.keys())[:5]  # Main credit factors
    scores = [component_scores[factor] for factor in factors]
    
    fig = px.bar(
        x=factors,
        y=scores,
        title="Credit Score Factor Breakdown",
        color=scores,
        color_continuous_scale="RdYlGn"
    )
    
    fig.update_layout(
        xaxis_title="Credit Factors",
        yaxis_title="Score (0-100)",
        showlegend=False
    )
    
    return fig

def create_loan_comparison_chart(loans_data):
    """Create loan comparison visualization"""
    loan_types = [loan['loan_type'] for loan in loans_data]
    interest_rates = [loan['interest_rate'] for loan in loans_data]
    max_amounts = [loan['max_amount']/100000 for loan in loans_data]  # Convert to lakhs
    approval_probs = [loan['approval_probability'] for loan in loans_data]
    
    fig = go.Figure()
    
    # Interest rates
    fig.add_trace(go.Bar(
        name='Interest Rate (%)',
        x=loan_types,
        y=interest_rates,
        yaxis='y',
        offsetgroup=1,
        marker_color='lightcoral'
    ))
    
    # Max amounts (in lakhs)
    fig.add_trace(go.Bar(
        name='Max Amount (₹ Lakhs)',
        x=loan_types,
        y=max_amounts,
        yaxis='y2',
        offsetgroup=2,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Loan Options Comparison',
        xaxis=dict(title='Loan Types'),
        yaxis=dict(title='Interest Rate (%)', side='left'),
        yaxis2=dict(title='Max Amount (₹ Lakhs)', side='right', overlaying='y'),
        barmode='group',
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Credit Score Analyzer</h1>
        <p>Advanced credit scoring with personalized insights and loan recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer and loan predictor
    analyzer = CreditScoreAnalyzer()
    loan_predictor = LoanPredictor()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Your Financial Profile")
        
        # Personal Information
        st.subheader("Personal Details")
        age = st.slider("Age", 18, 80, 30)
        annual_income = st.number_input("Annual Income (₹)", 100000, 10000000, 500000, step=50000)
        employment_type = st.selectbox("Employment Type", [
            'Salaried (Private)', 'Salaried (Government)', 'Self-employed', 
            'Business owner', 'Retired', 'Student'
        ])
        
        # Credit Information
        st.subheader("Credit History")
        credit_history_years = st.slider("Credit History (Years)", 0, 30, 5)
        payment_history = st.selectbox("Payment History", [
            'Always on time',
            'Mostly on time (1-2 late payments/year)',
            'Sometimes late (3-5 late payments/year)',
            'Frequently late (6+ late payments/year)',
            'Defaults/Charge-offs'
        ])
        
        credit_utilization = st.slider("Credit Utilization (%)", 0, 100, 30)
        credit_types = st.selectbox("Types of Credit Accounts", [1, 2, 3, 4, 5], index=2)
        recent_inquiries = st.slider("Hard Inquiries (Last 2 years)", 0, 10, 2)
        
        # Additional factors
        st.subheader("Additional Information")
        existing_loans = st.multiselect("Existing Loans", [
            'Home Loan', 'Car Loan', 'Personal Loan', 'Education Loan', 'Credit Card EMI'
        ])
        
        banking_relationship = st.slider("Banking Relationship (Years)", 0, 30, 5)
        
        generate_button = st.button("🔍 Generate Credit Score", type="primary")
    
    # Main content area
    if generate_button:
        # Prepare user data
        user_data = {
            'age': age,
            'annual_income': annual_income,
            'employment_type': employment_type,
            'credit_history_years': credit_history_years,
            'payment_history': payment_history,
            'credit_utilization': credit_utilization,
            'credit_types': credit_types,
            'recent_inquiries': recent_inquiries,
            'existing_loans': len(existing_loans),
            'banking_relationship': banking_relationship
        }
        
        # Show loading animation
        with st.spinner('🤖 AI is analyzing your credit profile...'):
            time.sleep(2)  # Simulate processing time
        
        # Calculate score
        score, component_scores = analyzer.calculate_score(user_data)
        category, color = analyzer.get_score_category(score)
        
        # Display results
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Score gauge
            fig_gauge = create_score_gauge(score)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.metric("Your Score", score, delta=score-650)
            st.metric("Category", category)
            st.metric("Percentile", f"{min(99, max(1, int((score-300)/600*100)))}%")

        
        with col3:
            st.metric("Max Possible", "900")
            st.metric("Average Score", "650")
            st.metric("Your Rank", category)
        
        # Factor breakdown
        st.subheader("📈 Score Factor Analysis")
        fig_breakdown = create_factor_breakdown(component_scores, analyzer.weights)
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Detailed insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Personalized Insights")
            insights = analyzer.generate_insights(score, user_data, component_scores)
            
            for insight in insights:
                if insight['type'] == 'warning':
                    st.markdown(f"""
                    <div class="warning-tip">
                        <strong>⚠️ {insight['title']}</strong><br>
                        {insight['message']}<br>
                        <small>Impact: {insight['impact']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="improvement-tip">
                        <strong>💡 {insight['title']}</strong><br>
                        {insight['message']}<br>
                        <small>Impact: {insight['impact']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Score Components")
            
            # Create a detailed breakdown table
            breakdown_data = {
                'Factor': ['Payment History', 'Credit Utilization', 'Credit History Length', 'Credit Mix', 'New Credit'],
                'Weight': ['35%', '30%', '15%', '10%', '10%'],
                'Your Score': [
                    f"{component_scores['payment_history']}/100",
                    f"{component_scores['credit_utilization']}/100",
                    f"{component_scores['credit_history_length']}/100",
                    f"{component_scores['credit_mix']}/100",
                    f"{component_scores['new_credit']}/100"
                ],
                'Impact': [
                    f"+{int(component_scores['payment_history'] * 0.35 * 6)}",
                    f"+{int(component_scores['credit_utilization'] * 0.30 * 6)}",
                    f"+{int(component_scores['credit_history_length'] * 0.15 * 6)}",
                    f"+{int(component_scores['credit_mix'] * 0.10 * 6)}",
                    f"+{int(component_scores['new_credit'] * 0.10 * 6)}"
                ]
            }
            
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True)
        
        # Action plan
        st.subheader("🚀 30-Day Action Plan")
        action_plan = []
        
        if component_scores['payment_history'] < 80:
            action_plan.append("Week 1-2: Set up automatic payments for all bills")
        
        if user_data['credit_utilization'] > 30:
            action_plan.append("Week 1-4: Pay down credit card balances to under 30%")
        
        if component_scores['credit_mix'] < 70:
            action_plan.append("Week 3-4: Consider diversifying credit types")
        
        action_plan.append("Ongoing: Monitor credit report monthly for errors")
        action_plan.append("Ongoing: Avoid new credit applications unless necessary")
        
        for i, action in enumerate(action_plan, 1):
            st.write(f"{i}. {action}")
        
        # LOAN PREDICTION SECTION
        st.markdown("---")
        st.header("💰 Personalized Loan Recommendations")
        
        # Get eligible loans
        eligible_loans = loan_predictor.get_eligible_loans(user_data, score)
        
        if eligible_loans:
            # Loan comparison chart
            fig_loans = create_loan_comparison_chart(eligible_loans)
            st.plotly_chart(fig_loans, use_container_width=True)
            
            # Loan details tabs
            loan_tabs = st.tabs([loan['loan_type'] for loan in eligible_loans])
            
            for i, loan in enumerate(eligible_loans):
                with loan_tabs[i]:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Interest Rate", f"{loan['interest_rate']}%")
                    with col2:
                        st.metric("Max Amount", f"₹{loan['max_amount']:,.0f}")
                    with col3:
                        st.metric("Max Tenure", f"{loan['max_tenure']} years" if loan['max_tenure'] > 0 else "Revolving")
                    with col4:
                        st.metric("Approval Probability", f"{loan['approval_probability']}%")
                    
                    # EMI Calculator
                    if loan['tenures']:
                        st.subheader("EMI Calculator")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            loan_amount = st.slider(
                                f"Loan Amount (₹)",
                                min_value=100000,
                                max_value=int(loan['max_amount']),
                                value=min(500000, int(loan['max_amount'])),
                                step=50000,
                                key=f"amount_{i}"
                            )
                        
                        with col2:
                            selected_tenure = st.selectbox(
                                "Tenure (Years)",
                                options=[t['tenure'] for t in loan['tenures']],
                                key=f"tenure_{i}"
                            )
                        
                        # Calculate EMI for selected amount and tenure
                        monthly_rate = loan['interest_rate'] / (12 * 100)
                        total_months = selected_tenure * 12
                        
                        if monthly_rate > 0:
                            emi = (loan_amount * monthly_rate * (1 + monthly_rate)**total_months) / \
                                  ((1 + monthly_rate)**total_months - 1)
                        else:
                            emi = loan_amount / total_months
                        
                        total_amount = emi * total_months
                        total_interest = total_amount - loan_amount
                        
                        # Display EMI details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"**Monthly EMI:** ₹{emi:,.0f}")
                        with col2:
                            st.info(f"**Total Amount:** ₹{total_amount:,.0f}")
                        with col3:
                            st.info(f"**Total Interest:** ₹{total_interest:,.0f}")
                    
                    # Bank offers
                    st.subheader("Best Bank Offers")
                    bank_data = []
                    for offer in loan['bank_offers']:
                        bank_data.append({
                            'Bank': offer['bank'],
                            'Interest Rate': f"{offer['rate']}%",
                            'Max Amount': f"₹{offer['max_amount']:,.0f}",
                            'Processing Fee': f"₹{offer['processing_fee']:,.0f}"
                        })
                    
                    df_banks = pd.DataFrame(bank_data)
                    st.dataframe(df_banks, use_container_width=True)
                    
                    # Loan-specific advice
                    st.subheader("💡 Loan-Specific Tips")
                    loan_tips = {
                        'Personal Loan': [
                            "Use for emergencies or debt consolidation",
                            "No collateral required but higher interest rates",
                            "Pay off early to save on interest"
                        ],
                        'Home Loan': [
                            "Lowest interest rates due to property collateral",
                            "Get tax benefits under Section 80C and 24(b)",
                            "Compare rates across multiple banks"
                        ],
                        'Car Loan': [
                            "Vehicle acts as collateral",
                            "Consider down payment to reduce EMI",
                            "Check for manufacturer tie-ups for better rates"
                        ],
                        'Education Loan': [
                            "No EMI during study period + 1 year",
                            "Tax deduction available under Section 80E",
                            "Consider government schemes for better rates"
                        ],
                        'Business Loan': [
                            "Maintain good business credit history",
                            "Keep financial statements updated",
                            "Consider collateral-free options for smaller amounts"
                        ],
                        'Gold Loan': [
                            "Quick approval and disbursal",
                            "Lower interest than personal loans",
                            "Gold remains safe in bank custody"
                        ],
                        'Credit Card': [
                            "Pay full amount by due date to avoid interest",
                            "Use for rewards and cash back",
                            "Maintain low credit utilization ratio"
                        ]
                    }
                    
                    tips = loan_tips.get(loan['loan_type'], ['Consult with bank for personalized advice'])
                    for tip in tips:
                        st.write(f"• {tip}")
        
        else:
            st.warning("⚠️ Based on your current credit profile, you may need to improve your credit score to access better loan options. Focus on the improvement tips above.")
            
            st.subheader("Alternative Options")
            st.write("• **Secured Loans:** Consider gold or FD-backed loans")
            st.write("• **Co-signer:** Add a co-applicant with good credit")
            st.write("• **Improve Score:** Work on credit improvement for 3-6 months")
        
        # Loan eligibility summary
        if eligible_loans:
            st.markdown("---")
            st.subheader("📋 Loan Eligibility Summary")
            
            summary_data = []
            for loan in eligible_loans:
                summary_data.append({
                    'Loan Type': loan['loan_type'],
                    'Status': '✅ Eligible' if loan['approval_probability'] > 70 else '⚠️ Conditional',
                    'Interest Rate': f"{loan['interest_rate']}%",
                    'Max Amount': f"₹{loan['max_amount']/100000:.1f}L",
                    'Approval Chance': f"{loan['approval_probability']}%"
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
        
        # Disclaimer
        st.markdown("""
        ---
        **⚠️ Important Disclaimer:** This is a simulated credit score and loan prediction system for educational purposes only. 
        For actual CIBIL scores, visit official credit bureaus. For real loan applications, consult with banks directly.
        Interest rates and loan amounts are indicative and may vary based on bank policies and market conditions.
        """)

if __name__ == "__main__":
    main()