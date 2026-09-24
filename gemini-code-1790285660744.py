"""
CDC Provisional Natality Dashboard (2025)
A single-file Streamlit application designed for undergraduate business analytics students.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="CDC Provisional Natality 2025 Dashboard",
    page_icon="👶",
    layout="wide"
)

# State name to standard 2-letter postal abbreviation mapping for maps
US_STATE_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
    'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# Explicit chronological month ordering
MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Colorblind-safe visual palette
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'female': '#e377c2',
    'male': '#1f77b4',
    'neutral': '#7f7f7f',
    'heatmap_scale': 'Viridis',
    'map_scale': 'Blues'
}


# ==============================================================================
# 2. DATA PIPELINE & VALIDATION
# ==============================================================================
def resolve_data_path() -> Path:
    """Dynamically resolve the dataset file path across local and Cloud environments."""
    candidate_paths = [
        Path(__file__).parent / "Provisional_Natality_2025_CDC1.csv",
        Path(__file__).parent / "data" / "Provisional_Natality_2025_CDC1.csv",
        Path.cwd() / "Provisional_Natality_2025_CDC1.csv",
        Path.cwd() / "data" / "Provisional_Natality_2025_CDC1.csv"
    ]
    
    for path in candidate_paths:
        if path.exists():
            return path
            
    raise FileNotFoundError("Could not locate 'Provisional_Natality_2025_CDC1.csv' in expected directories.")


def validate_dataset(df: pd.DataFrame) -> bool:
    """Run data integrity validation checks."""
    required_cols = {'state_of_residence', 'month', 'month_code', 'year_code', 'sex_of_infant', 'births'}
    
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Data Validation Error: Missing required columns: {missing}")
        return False
        
    if (df['births'] < 0).any():
        st.error("Data Validation Error: Dataset contains negative birth values.")
        return False
        
    if df[list(required_cols)].isna().any().any():
        st.error("Data Validation Error: Dataset contains unexpected null values.")
        return False
        
    return True


@st.cache_data(show_spinner="Loading natality dataset...")
def load_natality_data() -> pd.DataFrame:
    """Load, validate, and preprocess CDC Natality dataset."""
    path = resolve_data_path()
    df = pd.read_csv(path)
    
    if not validate_dataset(df):
        st.stop()
        
    # Enforce chronological ordering for months
    df['month'] = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)
    
    # Map state standard abbreviations for US maps
    df['state_abbr'] = df['state_of_residence'].map(US_STATE_ABBR)
    
    return df


# ==============================================================================
# 3. HEADER & KPI METRICS COMPONENTS
# ==============================================================================
def render_header():
    """Render title header, source attribution, and statistical notices."""
    st.title("👶 CDC Provisional Natality Dashboard (2025)")
    st.caption(
        "Interactive analytics dashboard designed for undergraduate business analytics students to explore "
        "U.S. natality trends across geographic regions, months, and infant sexes."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📌 **Data Status**: Figures reflect **provisional** 2025 CDC records subject to final verification.")
    with col2:
        st.warning("⚠️ **Analytical Note**: Values represent total **birth counts**, not **birth rates**. They do not control for total population size.")
        
    st.markdown("[🔗 Source: CDC National Center for Health Statistics (NCHS)](https://www.cdc.gov/nchs/)")
    st.divider()


def render_kpi_cards(filtered_df: pd.DataFrame):
    """Render top-level summary metrics cards."""
    if filtered_df.empty:
        return

    total_births = filtered_df['births'].sum()
    num_states = filtered_df['state_of_residence'].nunique()
    num_months = filtered_df['month'].nunique()
    
    avg_monthly_births = total_births / num_months if num_months > 0 else 0
    
    # Highest state calculation
    state_totals = filtered_df.groupby('state_of_residence')['births'].sum()
    top_state = state_totals.idxmax() if not state_totals.empty else "N/A"
    top_state_val = state_totals.max() if not state_totals.empty else 0
    
    # Highest month calculation
    month_totals = filtered_df.groupby('month', observed=False)['births'].sum()
    top_month = month_totals.idxmax() if not month_totals.empty else "N/A"
    top_month_val = month_totals.max() if not month_totals.empty else 0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Total Births", f"{total_births:,}")
    kpi2.metric("Geographies", f"{num_states:,}")
    kpi3.metric("Avg / Month", f"{int(round(avg_monthly_births)):,}")
    kpi4.metric("Top Geography", f"{top_state}", f"{top_state_val:,} births")
    kpi5.metric("Peak Month", f"{top_month}", f"{top_month_val:,} births")
    
    st.divider()


def render_empty_state_message():
    """Display warning banner when filters return zero rows."""
    st.warning("🔍 **No observations match your current filter selection.** Please expand your sidebar filters.")


# ==============================================================================
# 4. DASHBOARD TAB VIEWS
# ==============================================================================
def render_tab_overview(filtered_df: pd.DataFrame):
    """Render Overview Tab."""
    st.subheader("📈 Executive Overview")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Monthly Birth Trajectory")
        monthly_trend = filtered_df.groupby('month', observed=False)['births'].sum().reset_index()
        
        fig_trend = px.line(
            monthly_trend,
            x='month',
            y='births',
            markers=True,
            title="Total Births Across Selected Months",
            labels={'month': 'Month', 'births': 'Birth Count'},
            color_discrete_sequence=[COLOR_PALETTE['primary']]
        )
        fig_trend.update_layout(yaxis_range=[0, monthly_trend['births'].max() * 1.15])
        fig_trend.update_traces(hovertemplate="<b>%{x}</b><br>Births: %{y:,}<extra></extra>")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col2:
        st.markdown("#### Top vs. Bottom Geographies")
        top_n = st.slider("Select Top/Bottom N states", min_value=3, max_value=10, value=5)
        
        state_totals = filtered_df.groupby('state_of_residence')['births'].sum().reset_index()
        state_totals = state_totals.sort_values(by='births', ascending=False)
        
        if len(state_totals) >= top_n * 2:
            top_states = state_totals.head(top_n).copy()
            top_states['Category'] = f'Top {top_n}'
            bottom_states = state_totals.tail(top_n).copy()
            bottom_states['Category'] = f'Bottom {top_n}'
            combined = pd.concat([top_states, bottom_states])
        else:
            combined = state_totals.copy()
            combined['Category'] = 'All Selected'
            
        fig_comp = px.bar(
            combined,
            x='births',
            y='state_of_residence',
            color='Category',
            orientation='h',
            title=f"Top {top_n} vs. Bottom {top_n} Geography Volume",
            labels={'births': 'Total Births', 'state_of_residence': 'Geography'},
            color_discrete_map={f'Top {top_n}': COLOR_PALETTE['primary'], f'Bottom {top_n}': COLOR_PALETTE['secondary']}
        )
        fig_comp.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_range=[0, combined['births'].max() * 1.15])
        fig_comp.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,}<extra></extra>")
        st.plotly_chart(fig_comp, use_container_width=True)


def render_tab_geographic(filtered_df: pd.DataFrame):
    """Render Geographic Analysis Tab."""
    st.subheader("🗺️ Geographic Spatial & Density Analysis")
    
    st.markdown("#### U.S. State Birth Distribution Map")
    state_agg = filtered_df.groupby(['state_of_residence', 'state_abbr'])['births'].sum().reset_index()
    
    fig_map = px.choropleth(
        state_agg,
        locations='state_abbr',
        locationmode="USA-states",
        color='births',
        scope="usa",
        color_continuous_scale=COLOR_PALETTE['map_scale'],
        title="Total Birth Counts by State",
        labels={'births': 'Total Births', 'state_abbr': 'State'}
    )
    fig_map.update_traces(hovertemplate="<b>%{location}</b><br>Births: %{z:,}<extra></extra>")
    fig_map.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### State Volume Ranking")
        sorted_states = state_agg.sort_values(by='births', ascending=True)
        fig_rank = px.bar(
            sorted_states,
            x='births',
            y='state_of_residence',
            orientation='h',
            title="Geography Volume Ranking",
            labels={'births': 'Birth Count', 'state_of_residence': 'State'},
            color_discrete_sequence=[COLOR_PALETTE['primary']]
        )
        fig_rank.update_layout(height=max(400, len(sorted_states) * 20), xaxis_range=[0, sorted_states['births'].max() * 1.15])
        fig_rank.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,}<extra></extra>")
        st.plotly_chart(fig_rank, use_container_width=True)
        
    with col2:
        st.markdown("#### State-by-Month Heatmap")
        heatmap_data = filtered_df.groupby(['state_of_residence', 'month'], observed=False)['births'].sum().unstack(fill_value=0)
        
        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Month", y="Geography", color="Births"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=COLOR_PALETTE['heatmap_scale'],
            aspect="auto",
            title="State-by-Month Birth Density Matrix"
        )
        fig_heat.update_layout(height=max(400, len(heatmap_data) * 20))
        st.plotly_chart(fig_heat, use_container_width=True)


def render_tab_monthly_sex(filtered_df: pd.DataFrame):
    """Render Monthly and Sex Analysis Tab."""
    st.subheader("👫 Monthly & Infant-Sex Disaggregation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Monthly Trend by Infant Sex")
        sex_monthly = filtered_df.groupby(['month', 'sex_of_infant'], observed=False)['births'].sum().reset_index()
        
        fig_sex_line = px.line(
            sex_monthly,
            x='month',
            y='births',
            color='sex_of_infant',
            markers=True,
            title="Monthly Birth Trends by Infant Sex",
            labels={'month': 'Month', 'births': 'Birth Count', 'sex_of_infant': 'Infant Sex'},
            color_discrete_map={'Female': COLOR_PALETTE['female'], 'Male': COLOR_PALETTE['male']}
        )
        fig_sex_line.update_layout(yaxis_range=[0, sex_monthly['births'].max() * 1.15])
        fig_sex_line.update_traces(hovertemplate="<b>%{x} (%{fullData.name})</b><br>Births: %{y:,}<extra></extra>")
        st.plotly_chart(fig_sex_line, use_container_width=True)
        
    with col2:
        st.markdown("#### Sex Distribution Comparison")
        sex_totals = filtered_df.groupby('sex_of_infant')['births'].sum().reset_index()
        total_b = sex_totals['births'].sum()
        sex_totals['Percentage'] = (sex_totals['births'] / total_b * 100).round(2) if total_b > 0 else 0
        
        fig_sex_bar = px.bar(
            sex_totals,
            x='sex_of_infant',
            y='births',
            color='sex_of_infant',
            text=sex_totals['Percentage'].apply(lambda x: f"{x:.1f}%"),
            title="Total Birth Volume & Share by Sex",
            labels={'sex_of_infant': 'Infant Sex', 'births': 'Total Births'},
            color_discrete_map={'Female': COLOR_PALETTE['female'], 'Male': COLOR_PALETTE['male']}
        )
        fig_sex_bar.update_layout(yaxis_range=[0, sex_totals['births'].max() * 1.2])
        fig_sex_bar.update_traces(textposition='outside', hovertemplate="<b>%{x}</b><br>Births: %{y:,}<br>Share: %{text}<extra></extra>")
        st.plotly_chart(fig_sex_bar, use_container_width=True)


def render_tab_data_table(filtered_df: pd.DataFrame):
    """Render Data Table and CSV Download Tab."""
    st.subheader("📋 Searchable Filtered Data Table")
    st.caption("Inspect individual observations matching your current active sidebar filters.")
    
    search_term = st.text_input("🔍 Search within active filtered results (by state name or month):", "")
    
    display_df = filtered_df.copy()
    if search_term:
        mask = (
            display_df['state_of_residence'].str.contains(search_term, case=False, na=False) |
            display_df['month'].astype(str).str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]
        
    display_cols = ['state_of_residence', 'month', 'month_code', 'year_code', 'sex_of_infant', 'births']
    formatted_df = display_df[display_cols].sort_values(by=['state_of_residence', 'month_code'])
    
    st.dataframe(
        formatted_df.style.format({'births': '{:,}'}),
        use_container_width=True,
        hide_index=True
    )
    
    csv_data = formatted_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="cdc_filtered_natality_2025.csv",
        mime="text/csv",
        type="primary"
    )


def render_tab_about():
    """Render About the Data Tab."""
    st.subheader("📚 About the CDC Provisional Natality Dataset")
    
    st.markdown("""
    ### Data Source & Background
    * **Source**: Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)[cite: 1].
    * **Dataset**: Provisional Natality Records for 2025[cite: 1].
    * **Coverage**: All 50 U.S. States plus the District of Columbia[cite: 1].
    
    ---
    
    ### Key Analytical Concepts for Business Analytics Students
    
    #### 1. Raw Birth Counts vs. Crude Birth Rates
    * **Birth Count**: The absolute tally of registered live births in a given geography and timeframe[cite: 1].
    * **Crude Birth Rate (CBR)**: Expressed as $\\frac{\\text{Total Births}}{\\text{Total Population}} \\times 1,000$.
    * **Takeaway**: Highly populated states (e.g., California, Texas) naturally exhibit higher **birth counts** simply due to population scale. To measure fertility rates accurately across states, analysts must normalize counts by total population.
    
    #### 2. Provisional Data Considerations
    * Provisional records reflect early data submissions from state vital statistics offices[cite: 1].
    * Figures are subject to minor revisions prior to the release of official annual vital statistics reports[cite: 1].
    
    #### 3. Expected Natural Sex Ratio
    * Statistically, human birth sex ratios naturally favor male births slightly (typically $\\sim 105$ males per $100$ females) across large population samples.
    """)


# ==============================================================================
# 5. MAIN CONTROLLER & APPLICATION ENTRY POINT
# ==============================================================================
def init_session_state(all_states, all_months):
    """Initialize session state variables for sidebar state control."""
    if 'selected_states' not in st.session_state:
        st.session_state.selected_states = list(all_states)
    if 'selected_months' not in st.session_state:
        st.session_state.selected_months = list(all_months)
    if 'selected_sex' not in st.session_state:
        st.session_state.selected_sex = 'All'


def main():
    # 1. Load Data
    raw_df = load_natality_data()
    all_states = sorted(raw_df['state_of_residence'].unique().tolist())
    all_months = MONTH_ORDER
    
    init_session_state(all_states, all_months)
    
    # 2. Render Header
    render_header()
    
    # 3. Render Sidebar Controls
    st.sidebar.title("🎛️ Dashboard Controls")
    
    col_btn1, col_btn2 = st.sidebar.columns(2)
    
    if col_btn1.button("Select All", use_container_width=True):
        st.session_state.selected_states = list(all_states)
        st.session_state.selected_months = list(all_months)
        st.session_state.selected_sex = 'All'
        st.rerun()
        
    if col_btn2.button("Reset Filters", use_container_width=True):
        st.session_state.selected_states = list(all_states)
        st.session_state.selected_months = list(all_months)
        st.session_state.selected_sex = 'All'
        st.rerun()
        
    st.sidebar.divider()
    
    st.session_state.selected_states = st.sidebar.multiselect(
        "Select Geography / State(s):",
        options=all_states,
        default=st.session_state.selected_states
    )
    
    st.session_state.selected_months = st.sidebar.multiselect(
        "Select Month(s):",
        options=all_months,
        default=st.session_state.selected_months
    )
    
    sex_options = ['All', 'Female', 'Male']
    st.session_state.selected_sex = st.sidebar.radio(
        "Select Infant Sex:",
        options=sex_options,
        index=sex_options.index(st.session_state.selected_sex)
    )
    
    # Active Filter Summary
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Active Filter Summary")
    st.sidebar.write(f"• **Geographies**: {len(st.session_state.selected_states)} / {len(all_states)}")
    st.sidebar.write(f"• **Months**: {len(st.session_state.selected_months)} / 12")
    st.sidebar.write(f"• **Infant Sex**: {st.session_state.selected_sex}")
    
    # 4. Filter Dataset
    filtered_df = raw_df[
        (raw_df['state_of_residence'].isin(st.session_state.selected_states)) &
        (raw_df['month'].isin(st.session_state.selected_months))
    ]
    
    if st.session_state.selected_sex != 'All':
        filtered_df = filtered_df[filtered_df['sex_of_infant'] == st.session_state.selected_sex]
        
    # 5. Check for Empty State
    if filtered_df.empty:
        render_empty_state_message()
        return

    # 6. Render KPI Cards
    render_kpi_cards(filtered_df)
    
    # 7. Render Dashboard Tabs
    tab_overview, tab_geo, tab_monthly, tab_table, tab_about = st.tabs([
        "📊 Overview",
        "🗺️ Geographic Analysis",
        "📈 Monthly & Sex Analysis",
        "📋 Data Table & Download",
        "ℹ️ About the Data"
    ])
    
    with tab_overview:
        render_tab_overview(filtered_df)
        
    with tab_geo:
        render_tab_geographic(filtered_df)
        
    with tab_monthly:
        render_tab_monthly_sex(filtered_df)
        
    with tab_table:
        render_tab_data_table(filtered_df)
        
    with tab_about:
        render_tab_about()


if __name__ == "__main__":
    main()