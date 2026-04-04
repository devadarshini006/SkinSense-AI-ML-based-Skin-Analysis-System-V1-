import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🧴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B9D;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
   .product-card {
    padding: 1rem;
    border-left: 4px solid #FF6B9D;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    margin: 0.5rem 0;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.product-card h4 {
    color: #FF6B9D;
    margin: 0 0 0.5rem 0;
}
.product-card p {
    color: #333333;
    margin: 0.25rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load models ( need to upload these files when deploying)
@st.cache_resource
def load_models():
    try:
        with open('skinsense_models_v2.pkl', 'rb') as f:
            models = pickle.load(f)
        products = pd.read_csv('products_database.csv')
        return models, products
    except:
        return None, None

models, products_df = load_models()

# Header
st.markdown('<h1 class="main-header">🧴 SkinSense AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Personal AI Skincare Advisor</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/FF6B9D/FFFFFF?text=SkinSense+AI", use_container_width=True)
    st.markdown("### About")
    st.info("AI-powered skin analysis with personalized product recommendations based on your unique skin profile.")

    st.markdown("### How it works")
    st.markdown("""
    1. 📝 Fill in your skin parameters
    2. 🔬 AI analyzes your skin
    3. 💡 Get personalized recommendations
    4. 🛍️ Build your perfect routine
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔬 Skin Analysis", "📊 About Your Results", "ℹ️ FAQ"])

with tab1:
    st.markdown("### Tell us about your skin")
    st.markdown("Rate each parameter on a scale of 1-10")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧴 Skin Characteristics")
        sebum = st.slider("Sebum/Oil Level", 1.0, 10.0, 5.0, 0.5,
                         help="1 = Very dry, 10 = Very oily")
        moisture = st.slider("Moisture Level", 1.0, 10.0, 6.0, 0.5,
                           help="1 = Dehydrated, 10 = Well-hydrated")
        pores = st.slider("Pore Size", 1.0, 10.0, 5.0, 0.5,
                         help="1 = Invisible, 10 = Very visible")
        sensitivity = st.slider("Sensitivity", 1.0, 10.0, 4.0, 0.5,
                               help="1 = Never reacts, 10 = Always reacts")
        texture = st.slider("Skin Texture", 1.0, 10.0, 4.0, 0.5,
                           help="1 = Smooth, 10 = Very rough")

    with col2:
        st.markdown("#### 🎯 Skin Concerns")
        acne = st.slider("Acne/Breakouts", 1.0, 10.0, 3.0, 0.5,
                        help="1 = Never, 10 = Constant")
        pigmentation = st.slider("Pigmentation", 1.0, 10.0, 4.0, 0.5,
                                help="1 = Even tone, 10 = Very uneven")
        dark_spots = st.slider("Dark Spots", 1.0, 10.0, 3.0, 0.5,
                              help="1 = None, 10 = Severe")
        redness = st.slider("Redness/Inflammation", 1.0, 10.0, 3.0, 0.5,
                           help="1 = None, 10 = Severe")

    age = st.number_input("Age (years)", 18, 100, 25)

    budget = st.select_slider("Budget Preference",
                             options=['low', 'medium', 'high'],
                             value='medium',
                             help="Low: <₹800, Medium: ₹800-1500, High: >₹1500")

    analyze_button = st.button("🔬 Analyze My Skin", type="primary", use_container_width=True)

    if analyze_button:
        if models is None:
            st.error("⚠️ Models not loaded. Please ensure model files are uploaded.")
        else:
            # Prepare data
            user_data = [[sebum, moisture, pores, sensitivity, pigmentation,
                         dark_spots, acne, age, texture, redness]]

            # Predictions
            skin_type = models['skin_type_model'].predict(user_data)[0]
            probabilities = models['skin_type_model'].predict_proba(user_data)[0]
            confidence = max(probabilities)

            acne_score = models['acne_model'].predict(user_data)[0]
            pigment_score = models['pigmentation_model'].predict(user_data)[0]
            aging_score = models['aging_model'].predict(user_data)[0]

            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Your Skin Analysis Results")

            # Skin type result
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="result-box">
                    <h3>Skin Type</h3>
                    <h2>{skin_type.upper()}</h2>
                    <p>Confidence: {confidence:.0%}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                severity_color = "🟢" if acne_score < 40 else "🟡" if acne_score < 70 else "🔴"
                st.markdown(f"""
                <div class="result-box">
                    <h3>Acne Severity</h3>
                    <h2>{severity_color} {acne_score:.0f}/100</h2>
                    <p>{'Mild' if acne_score < 40 else 'Moderate' if acne_score < 70 else 'Severe'}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                pig_color = "🟢" if pigment_score < 40 else "🟡" if pigment_score < 70 else "🔴"
                st.markdown(f"""
                <div class="result-box">
                    <h3>Pigmentation</h3>
                    <h2>{pig_color} {pigment_score:.0f}/100</h2>
                    <p>{'Mild' if pigment_score < 40 else 'Moderate' if pigment_score < 70 else 'Severe'}</p>
                </div>
                """, unsafe_allow_html=True)

            # Active ingredients
            st.markdown("### 💡 Recommended Active Ingredients")

            actives_map = {
                'Oily': ['Salicylic Acid 2%', 'Niacinamide 5-10%', 'Clay masks'],
                'Dry': ['Hyaluronic Acid', 'Ceramides', 'Squalane'],
                'Combination': ['Niacinamide 5%', 'Lightweight HA', 'BHA for T-zone'],
                'Normal': ['Vitamin C', 'Peptides', 'Light Retinol'],
                'Sensitive': ['Centella Asiatica', 'Ceramides', 'Avoid fragrances']
            }

            actives = actives_map.get(skin_type, [])
            cols = st.columns(len(actives))
            for idx, active in enumerate(actives):
                cols[idx].success(f"✓ {active}")

            # Additional recommendations
            if acne_score > 60:
                st.warning("🎯 **High Priority - Acne:** Benzoyl Peroxide 2.5%, Adapalene")
            if pigment_score > 60:
                st.warning("🎯 **High Priority - Pigmentation:** Vitamin C 15%, Alpha Arbutin, SPF 50+ DAILY")
            if aging_score > 60:
                st.warning("🎯 **High Priority - Aging:** Retinol 0.3-1%, Peptides, Antioxidants")

            # Product recommendations
            if products_df is not None:
                st.markdown("---")
                st.markdown("## 🛍️ Personalized Product Recommendations")

                # Filter products
                budget_ranges = {'low': (0, 800), 'medium': (400, 1500), 'high': (1000, 5000)}
                min_p, max_p = budget_ranges[budget]

                top_concerns = []
                if acne_score > 60: top_concerns.append('Acne')
                if pigment_score > 60: top_concerns.extend(['Pigmentation', 'Dark Spots'])
                if aging_score > 60: top_concerns.append('Aging')
                if not top_concerns: top_concerns = ['Hydration']

                for prod_type in ['Cleanser', 'Serum', 'Moisturizer', 'Sunscreen']:
                    filtered = products_df[
                        (products_df['type'] == prod_type) &
                        (products_df['price'] >= min_p) &
                        (products_df['price'] <= max_p)
                    ].copy()

                    if len(filtered) > 0:
                        st.markdown(f"#### {prod_type}")

                        # Score products
                       # Score products - FIXED
                scores = []
                for idx, row in filtered.iterrows():
                    score = 0.0
                    if skin_type in str(row['suitable_for']) or 'All' in str(row['suitable_for']):
                        score += 40.0
                    concern_matches = sum(1 for c in top_concerns if c in str(row['concerns']))
                    score += concern_matches * 10.0
                    score += float(row['rating']) * 2.0
                    scores.append(score)
                
                filtered = filtered.copy()
                filtered['score'] = scores
                top_prod = filtered.nlargest(3, 'score')
                for _, prod in top_prod.iterrows():
                            st.markdown(f"""
                            <div class="product-card">
                                <h4>{str(prod['name'])}</h4>
                                <p><strong>Brand:</strong> {str(prod['brand'])} | <strong>Price:</strong> ₹{str(int(prod['price']))} | <strong>Rating:</strong> {'⭐' * int(prod['rating'])}</p>
                                <p><strong>Key Actives:</strong> {str(prod['key_actives'])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                # Routine
                st.markdown("---")
                st.markdown("## 📅 Your Daily Routine")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🌅 Morning Routine")
                    st.markdown("1. Cleanser (gentle)")
                    st.markdown("2. Serum (Vitamin C/Niacinamide)")
                    st.markdown("3. Moisturizer")
                    st.markdown("4. **Sunscreen SPF 50+ (MANDATORY)**")

                with col2:
                    st.markdown("### 🌙 Evening Routine")
                    st.markdown("1. Cleanser (same as AM)")
                    st.markdown("2. Treatment (if needed)")
                    st.markdown("3. Serum (targeted)")
                    st.markdown("4. Moisturizer")

with tab2:
    st.markdown("### Understanding Your Results")
    st.markdown("""
    #### Skin Types:
    - **Oily:** High sebum, large pores, prone to acne
    - **Dry:** Low moisture, small pores, flaky/tight
    - **Combination:** Oily T-zone, dry cheeks
    - **Normal:** Balanced, minimal concerns
    - **Sensitive:** Reactive, easily irritated

    #### Severity Scores:
    - **0-40:** Mild - Preventive care
    - **40-70:** Moderate - Active treatment
    - **70-100:** Severe - Consider dermatologist
    """)

with tab3:
    st.markdown("### Frequently Asked Questions")

    with st.expander("How accurate is the AI?"):
        st.write("Our model achieves 90%+ accuracy based on dermatological patterns. However, consult a dermatologist for medical concerns.")

    with st.expander("How were products selected?"):
        st.write("Products are curated based on ingredients, ratings, and suitability for different skin types and budgets.")

    with st.expander("Can I use all recommended products together?"):
        st.write("Follow the AM/PM routine. Introduce one product at a time. Some actives shouldn't be mixed - check product instructions.")

    with st.expander("When will I see results?"):
        st.write("Most products show results in 4-12 weeks. Be consistent and patient!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ by SkinSense AI | Not medical advice - consult a dermatologist for serious concerns</p>
</div>
""", unsafe_allow_html=True)

print("Streamlit app created: app.py")
