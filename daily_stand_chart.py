"""
Daily Stand Count Visualization
Minimal, clinical bar chart showing stands per calendar day
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from collections import defaultdict
from typing import List, Dict, Optional
import numpy as np
import time


def get_stand_transitions(event_logger) -> List[float]:
    """
    Extract timestamps of posture transitions to STANDING.

    Args:
        event_logger: EventLogger instance

    Returns:
        List of timestamps when posture changed to STANDING
    """
    stand_timestamps = []

    # Get all events from buffer
    events = list(event_logger.event_buffer)

    # Filter out timestamps before year 2000 (to exclude 1969 bugs and invalid data)
    # Unix timestamp for Jan 1, 2000 00:00:00 UTC is 946684800
    MIN_VALID_TIMESTAMP = 946684800

    for event in events:
        # Look for confirmed stand events
        event_type = event.get('event_type', '')

        if event_type in ['stand_confirmed', 'stand_false_positive']:
            # Check is_confirmed flag directly (most reliable)
            is_confirmed = event.get('is_confirmed', False)

            if is_confirmed:
                timestamp = event.get('timestamp')
                if timestamp is not None and timestamp >= MIN_VALID_TIMESTAMP:
                    stand_timestamps.append(timestamp)

    return stand_timestamps


def aggregate_stands_by_day(timestamps: List[float]) -> Dict[str, int]:
    """
    Aggregate stand counts per calendar day.

    Args:
        timestamps: List of Unix timestamps

    Returns:
        Dictionary mapping date strings (YYYY-MM-DD) to stand counts
    """
    daily_counts = defaultdict(int)

    for ts in timestamps:
        dt = datetime.fromtimestamp(ts)
        date_key = dt.strftime('%Y-%m-%d')
        daily_counts[date_key] += 1

    return dict(daily_counts)


def generate_mock_stand_data(days: int = 7) -> List[float]:
    """
    Generate mock stand timestamps for testing.

    Args:
        days: Number of days to generate data for

    Returns:
        List of mock timestamps
    """
    timestamps = []
    now = time_now = datetime.now()

    for day_offset in range(days):
        target_date = now - timedelta(days=days - day_offset - 1)

        # Random number of stands per day (5-20)
        num_stands = np.random.randint(5, 21)

        for _ in range(num_stands):
            # Random time during waking hours (6am - 10pm)
            hour = np.random.randint(6, 22)
            minute = np.random.randint(0, 60)

            stand_time = target_date.replace(hour=hour, minute=minute, second=0)
            timestamps.append(stand_time.timestamp())

    return sorted(timestamps)


def render_daily_stand_chart(event_logger, show_mock: bool = False) -> None:
    """
    Render a minimal, clinical bar chart of stands per day.

    Args:
        event_logger: EventLogger instance
        show_mock: If True, use mock data for demonstration
    """
    st.subheader("Stands per Day")

    # Get stand timestamps
    if show_mock:
        timestamps = generate_mock_stand_data(days=7)
        st.caption("Showing mock data for demonstration")
    else:
        timestamps = get_stand_transitions(event_logger)

    # Aggregate by day
    daily_counts = aggregate_stands_by_day(timestamps)

    if not daily_counts:
        st.info("No stand events recorded yet")
        return

    # Sort dates and prepare chart data
    sorted_dates = sorted(daily_counts.keys())
    counts = [daily_counts[d] for d in sorted_dates]

    # Validate counts data - ensure all values are positive integers
    counts = [int(c) if c is not None and c > 0 else 0 for c in counts]

    # If all counts are 0, show info message
    if sum(counts) == 0:
        st.info("No valid stand events recorded yet")
        return

    # Format dates for display (MM/DD)
    display_labels = [datetime.strptime(d, '%Y-%m-%d').strftime('%m/%d') for d in sorted_dates]

    # Create gradient colors for bars (teal to blue based on value)
    max_count = max(counts) if counts else 1
    colors = []
    for count in counts:
        # Interpolate based on count relative to max
        ratio = count / max_count if max_count > 0 else 0
        r = int(0x5A + (0x2D - 0x5A) * ratio)
        g = int(0x7C + (0x7A - 0x7C) * ratio)
        b = int(0x71 + (0x8F - 0x71) * ratio)
        colors.append(f'rgb({r},{g},{b})')

    # Create bar chart with minimal, clinical design
    fig = go.Figure(data=[
        go.Bar(
            x=display_labels,
            y=counts,
            marker_color=colors,  # Gradient colors
            hovertemplate='<b>%{x}</b><br>Stands: %{y}<extra></extra>'
        )
    ])

    # Minimal layout with clear axis labels
    fig.update_layout(
        title=None,
        height=350,
        margin=dict(l=60, r=20, t=20, b=60),
        plot_bgcolor='white',
        showlegend=False
    )

    # Clean, minimal axis styling
    fig.update_xaxes(
        title_text='Date',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71'
    )

    fig.update_yaxes(
        title_text='Number of Stands',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71',
        zeroline=True,
        zerolinecolor='#5A7C71'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show summary statistics below chart
    col1, col2, col3 = st.columns(3)

    with col1:
        total_stands = sum(counts)
        st.metric("Total Stands", total_stands)

    with col2:
        avg_stands = np.mean(counts) if counts else 0
        st.metric("Average per Day", f"{avg_stands:.1f}")

    with col3:
        max_stands = max(counts) if counts else 0
        max_day = display_labels[counts.index(max_stands)] if counts else "N/A"
        st.metric("Best Day", f"{max_stands} ({max_day})")


def aggregate_stands_by_weekday(timestamps: List[float]) -> Dict[str, int]:
    """
    Aggregate stand counts by weekday.

    Args:
        timestamps: List of Unix timestamps

    Returns:
        Dictionary mapping weekday names to stand counts
    """
    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = {day: 0 for day in weekday_labels}

    for ts in timestamps:
        dt = datetime.fromtimestamp(ts)
        weekday = weekday_labels[dt.weekday()]
        weekday_counts[weekday] += 1

    return weekday_counts


def render_weekly_activity_summary(event_logger, show_mock: bool = False) -> None:
    """
    Render a bar chart showing stand counts aggregated by weekday.

    Args:
        event_logger: EventLogger instance
        show_mock: If True, use mock data for demonstration
    """
    st.subheader("Weekly Standing Frequency")

    # Get stand timestamps
    if show_mock:
        timestamps = generate_mock_heatmap_data(days=28)
        st.caption("Showing mock data for demonstration")
    else:
        timestamps = get_stand_transitions(event_logger)

    if not timestamps:
        st.info("No stand events recorded yet")
        return

    # Aggregate by weekday
    weekday_counts = aggregate_stands_by_weekday(timestamps)

    # Preserve Monday-Sunday order
    weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    counts = [weekday_counts[day] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

    # Create gradient colors for bars (teal to blue)
    n_bars = 7
    colors = []
    for i in range(n_bars):
        # Interpolate between teal (#5A7C71) and blue (#2D7A8F)
        ratio = i / max(1, n_bars - 1)
        r = int(0x5A + (0x2D - 0x5A) * ratio)
        g = int(0x7C + (0x7A - 0x7C) * ratio)
        b = int(0x71 + (0x8F - 0x71) * ratio)
        colors.append(f'rgb({r},{g},{b})')

    # Create bar chart with minimal, clinical design
    fig = go.Figure(data=[
        go.Bar(
            x=weekday_order,
            y=counts,
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Stands: %{y}<extra></extra>'
        )
    ])

    # Minimal layout
    fig.update_layout(
        title=None,
        height=300,
        margin=dict(l=60, r=20, t=20, b=50),
        plot_bgcolor='white',
        showlegend=False
    )

    # Clean axis styling
    fig.update_xaxes(
        title_text='Day of Week',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71'
    )

    fig.update_yaxes(
        title_text='Number of Stands',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71',
        zeroline=True,
        zerolinecolor='#5A7C71'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    total_stands = sum(counts)
    avg_per_day = total_stands / 7 if total_stands > 0 else 0
    max_count = max(counts) if counts else 0
    max_day = weekday_order[counts.index(max_count)] if counts else "N/A"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Stands", total_stands)

    with col2:
        st.metric("Average per Day", f"{avg_per_day:.1f}")

    with col3:
        st.metric("Most Active Day", f"{max_day} ({max_count})")


def render_compact_daily_stand_chart(event_logger) -> None:
    """
    Render a compact version for embedding in other dashboards.

    Args:
        event_logger: EventLogger instance
    """
    timestamps = get_stand_transitions(event_logger)
    daily_counts = aggregate_stands_by_day(timestamps)

    if not daily_counts:
        st.caption("No stands recorded")
        return

    # Show last 7 days
    sorted_dates = sorted(daily_counts.keys())[-7:]
    counts = [daily_counts[d] for d in sorted_dates]

    display_labels = [datetime.strptime(d, '%Y-%m-%d').strftime('%a') for d in sorted_dates]

    # Create gradient colors for bars (teal to blue)
    n_bars = len(counts)
    colors = []
    for i in range(n_bars):
        # Interpolate between teal (#5A7C71) and blue (#2D7A8F)
        ratio = i / max(1, n_bars - 1)
        r = int(0x5A + (0x2D - 0x5A) * ratio)
        g = int(0x7C + (0x7A - 0x7C) * ratio)
        b = int(0x71 + (0x8F - 0x71) * ratio)
        colors.append(f'rgb({r},{g},{b})')

    # Compact bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=display_labels,
            y=counts,
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Stands: %{y}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=None,
        height=200,
        margin=dict(l=40, r=10, t=10, b=40),
        plot_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        title_text='Stands',
        showgrid=True,
        gridcolor='#E8F0F0'
    )

    st.plotly_chart(fig, use_container_width=True)


def aggregate_stands_by_hour_and_weekday(timestamps: List[float]) -> np.ndarray:
    """
    Aggregate stand counts by hour of day and day of week.

    Args:
        timestamps: List of Unix timestamps

    Returns:
        2D numpy array (7 days x 24 hours) with stand counts
        Days are ordered Monday=0 to Sunday=6
    """
    # Initialize 7x24 matrix (days x hours)
    heatmap_data = np.zeros((7, 24), dtype=int)

    for ts in timestamps:
        dt = datetime.fromtimestamp(ts)
        weekday = dt.weekday()  # Monday=0, Sunday=6
        hour = dt.hour

        heatmap_data[weekday, hour] += 1

    return heatmap_data


def generate_mock_heatmap_data(days: int = 28) -> List[float]:
    """
    Generate mock stand timestamps with realistic circadian patterns for heatmap.

    Args:
        days: Number of days to generate data for

    Returns:
        List of mock timestamps with realistic activity patterns
    """
    timestamps = []
    now = datetime.now()

    for day_offset in range(days):
        target_date = now - timedelta(days=days - day_offset - 1)
        weekday = target_date.weekday()

        # Define activity patterns by weekday
        # Weekdays: peak morning (8-10am) and afternoon (2-4pm)
        # Weekends: more spread out, later start (9-11am)
        if weekday < 5:  # Monday - Friday
            morning_peak = np.random.randint(7, 10)
            afternoon_peak = np.random.randint(13, 16)
            num_stands = np.random.randint(8, 18)
        else:  # Saturday - Sunday
            morning_peak = np.random.randint(8, 11)
            afternoon_peak = np.random.randint(14, 17)
            num_stands = np.random.randint(5, 15)

        for _ in range(num_stands):
            # Choose around peak times with some variance
            if np.random.random() < 0.5:
                base_hour = morning_peak
            else:
                base_hour = afternoon_peak

            hour = np.clip(base_hour + np.random.randint(-2, 3), 6, 20)
            minute = np.random.randint(0, 60)

            stand_time = target_date.replace(hour=hour, minute=minute, second=0)
            timestamps.append(stand_time.timestamp())

    return sorted(timestamps)


def render_stand_activity_heatmap(event_logger, show_mock: bool = False) -> None:
    """
    Render a heatmap showing stand activity by hour of day and day of week.

    X-axis: Hour (0-23)
    Y-axis: Weekday (Mon-Sun)
    Cell value: Number of stand events

    Uses Viridis perceptually uniform color scale.

    Args:
        event_logger: EventLogger instance
        show_mock: If True, use mock data for demonstration
    """
    st.subheader("Standing Activity Pattern")

    # Get stand timestamps
    if show_mock:
        timestamps = generate_mock_heatmap_data(days=28)
        st.caption("Showing mock data for demonstration")
    else:
        timestamps = get_stand_transitions(event_logger)

    if not timestamps:
        st.info("No stand events recorded yet")
        return

    # Aggregate by hour and weekday
    heatmap_data = aggregate_stands_by_hour_and_weekday(timestamps)

    # Day labels (Monday to Sunday)
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Hour labels (0-23)
    hour_labels = [str(h) for h in range(24)]

    # Create heatmap with teal-to-blue gradient
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=hour_labels,
        y=weekday_labels,
        colorscale=[
            [0.0, '#F0F9F4'],   # Light green/white for zero
            [0.2, '#C8E6D9'],
            [0.4, '#5A7C71'],   # Teal
            [0.6, '#4287A5'],
            [0.8, '#2D7A8F'],   # Blue
            [1.0, '#1A4D5E']    # Dark blue
        ],
        colorbar=dict(
            title='Stands',
            title_side='right',
            len=0.85,
            x=1.02
        ),
        hovertemplate='<b>%{y} %{x}:00</b><br>Stands: %{z}<extra></extra>'
    ))

    # Minimal layout
    fig.update_layout(
        title=None,
        height=350,
        margin=dict(l=60, r=80, t=20, b=50),
        plot_bgcolor='white'
    )

    # Clean axis styling
    fig.update_xaxes(
        title_text='Hour of Day',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71',
        tickmode='array',
        tickvals=hour_labels[::3],  # Show every 3rd hour label
        ticktext=[str(h) for h in range(24)][::3]
    )

    fig.update_yaxes(
        title_text='Day of Week',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show pattern insights
    total_stands = len(timestamps)

    # Calculate peak activity
    row_sums = np.sum(heatmap_data, axis=1)
    col_sums = np.sum(heatmap_data, axis=0)

    peak_day_idx = np.argmax(row_sums)
    peak_hour_idx = np.argmax(col_sums)

    peak_day = weekday_labels[peak_day_idx]
    peak_hour = peak_hour_idx

    # Calculate daytime vs nighttime activity
    daytime_stands = np.sum(heatmap_data[:, 6:18])  # 6am-6pm
    nighttime_stands = total_stands - daytime_stands

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Stands", total_stands)

    with col2:
        st.metric("Peak Day", peak_day)

    with col3:
        st.metric("Peak Hour", f"{peak_hour}:00")

    with col4:
        daytime_pct = (daytime_stands / total_stands * 100) if total_stands > 0 else 0
        st.metric("Daytime Activity", f"{daytime_pct:.0f}%")


def aggregate_stands_by_month(timestamps: List[float]) -> Dict[str, int]:
    """
    Aggregate stand counts by month.

    Args:
        timestamps: List of Unix timestamps

    Returns:
        Dictionary mapping month strings (YYYY-MM) to stand counts,
        sorted chronologically
    """
    monthly_counts = defaultdict(int)

    for ts in timestamps:
        dt = datetime.fromtimestamp(ts)
        month_key = dt.strftime('%Y-%m')
        monthly_counts[month_key] += 1

    # Sort by month
    sorted_months = dict(sorted(monthly_counts.items()))
    return sorted_months


def calculate_rolling_average(values: List[int], window: int = 3) -> List[float]:
    """
    Calculate rolling average to smooth trend data.

    Args:
        values: List of integer values
        window: Window size for rolling average

    Returns:
        List of smoothed values (same length as input)
    """
    if len(values) < window:
        return [float(v) for v in values]

    smoothed = []
    padded = list(values)  # No padding, use available data

    for i in range(len(padded)):
        # Use available data points (smaller window at edges)
        start_idx = max(0, i - window // 2)
        end_idx = min(len(padded), i + window // 2 + 1)
        window_values = padded[start_idx:end_idx]
        smoothed.append(sum(window_values) / len(window_values))

    return smoothed


def generate_mock_monthly_data(months: int = 12) -> List[float]:
    """
    Generate mock stand timestamps spanning multiple months.

    Args:
        months: Number of months to generate data for

    Returns:
        List of mock timestamps
    """
    timestamps = []
    now = datetime.now()

    for month_offset in range(months):
        # Calculate target month
        year = now.year
        month = now.month - month_offset
        while month <= 0:
            month += 12
            year -= 1

        # Days in this month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        else:
            # February (simplified, not accounting for leap years)
            days_in_month = 28

        # Base activity with some trend
        base_stands_per_day = 10 + month_offset * 0.5  # Slight upward trend
        base_stands_per_day += np.random.normal(0, 2)  # Add variance

        for day in range(1, days_in_month + 1):
            num_stands = int(max(5, base_stands_per_day + np.random.normal(0, 3)))

            for _ in range(num_stands):
                hour = np.random.randint(6, 22)
                minute = np.random.randint(0, 60)

                try:
                    stand_time = datetime(year, month, day, hour, minute)
                    timestamps.append(stand_time.timestamp())
                except ValueError:
                    pass  # Skip invalid dates (e.g., Feb 30)

    return sorted(timestamps)


def render_longitudinal_mobility_trend(event_logger, show_mock: bool = False) -> None:
    """
    Render a monthly trend line showing total stand events per month.

    X-axis: Month
    Y-axis: Total stand count

    Includes a rolling average line to smooth noise.

    Args:
        event_logger: EventLogger instance
        show_mock: If True, use mock data for demonstration
    """
    st.subheader("Longitudinal Mobility Trend")

    # Get stand timestamps
    if show_mock:
        timestamps = generate_mock_monthly_data(months=12)
        st.caption("Showing mock data for demonstration")
    else:
        timestamps = get_stand_transitions(event_logger)

    if not timestamps:
        st.info("No stand events recorded yet")
        return

    # Aggregate by month
    monthly_counts = aggregate_stands_by_month(timestamps)

    if not monthly_counts:
        st.info("No monthly data available")
        return

    # Extract month labels and counts
    months = list(monthly_counts.keys())
    counts = list(monthly_counts.values())

    # Format month labels for display (Jan 2024, Feb 2024, etc.)
    month_labels = []
    for m in months:
        dt = datetime.strptime(m, '%Y-%m')
        label = dt.strftime('%b %Y')
        month_labels.append(label)

    # Calculate rolling average
    rolling_avg = calculate_rolling_average(counts, window=3)

    # Create gradient colors for bars (teal to blue based on value)
    max_count = max(counts) if counts else 1
    colors = []
    for count in counts:
        # Interpolate based on count relative to max
        ratio = count / max_count if max_count > 0 else 0
        r = int(0x5A + (0x2D - 0x5A) * ratio)
        g = int(0x7C + (0x7A - 0x7C) * ratio)
        b = int(0x71 + (0x8F - 0x71) * ratio)
        colors.append(f'rgb({r},{g},{b})')

    # Create trend line chart
    fig = go.Figure()

    # Add bar chart for actual monthly counts with gradient
    fig.add_trace(go.Bar(
        x=month_labels,
        y=counts,
        name='Monthly Stands',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Stands: %{y}<extra></extra>'
    ))

    # Add rolling average line
    fig.add_trace(go.Scatter(
        x=month_labels,
        y=rolling_avg,
        name='Trend (3-month avg)',
        mode='lines',
        line=dict(color='#2D7A8F', width=3),
        hovertemplate='<b>%{x}</b><br>Trend: %{y:.1f}<extra></extra>'
    ))

    # Minimal layout
    fig.update_layout(
        title=None,
        height=350,
        margin=dict(l=60, r=20, t=40, b=60),
        plot_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified'
    )

    # Clean axis styling
    fig.update_xaxes(
        title_text='Month',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71'
    )

    fig.update_yaxes(
        title_text='Total Stand Count',
        showgrid=True,
        gridcolor='#E8F0F0',
        linewidth=1,
        linecolor='#5A7C71',
        zeroline=True,
        zerolinecolor='#5A7C71'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    total_months = len(months)
    total_stands = sum(counts)
    avg_per_month = total_stands / total_months if total_months > 0 else 0

    # Calculate trend direction
    if len(counts) >= 2:
        # Compare first half vs second half
        mid_point = len(counts) // 2
        earlier_avg = np.mean(counts[:mid_point]) if mid_point > 0 else counts[0]
        recent_avg = np.mean(counts[mid_point:]) if mid_point < len(counts) else counts[-1]
        trend_change = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0

        if trend_change > 5:
            trend_text = "Improving"
        elif trend_change < -5:
            trend_text = "Declining"
        else:
            trend_text = "Stable"
    else:
        trend_text = "Insufficient data"
        trend_change = 0

    # Find best and worst months
    best_month_idx = counts.index(max(counts)) if counts else 0
    worst_month_idx = counts.index(min(counts)) if counts else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Months Tracked", total_months)

    with col2:
        st.metric("Avg per Month", f"{avg_per_month:.0f}")

    with col3:
        st.metric("Trend", trend_text)

    with col4:
        best_month = month_labels[best_month_idx] if month_labels else "N/A"
        best_count = counts[best_month_idx] if counts else 0
        st.metric("Best Month", f"{best_month} ({best_count})")


if __name__ == "__main__":
    # Test the visualization standalone
    from event_logger import EventLogger
    import time

    st.set_page_config(page_title="Daily Stand Chart Demo", layout="wide")

    st.title("Daily Stand Count Visualization")

    # Initialize event logger
    if 'test_event_logger' not in st.session_state:
        test_logger = EventLogger(log_directory="./test_stand_logs", buffer_size=100)
        st.session_state.test_event_logger = test_logger

        # Add mock data
        now = time.time()
        for i in range(50):
            is_confirmed = np.random.random() > 0.15
            sbp_drop = np.clip(np.random.normal(16, 4), 5, 35)

            # Spread across 5 days
            timestamp = now - (5 - (i % 5)) * 86400 - np.random.randint(0, 43200)

            test_logger.log_stand_event(
                is_confirmed=is_confirmed,
                timestamp=timestamp,
                pitch_angle=70.0,
                sbp_drop=sbp_drop,
                posture_state="Standing",
                bp_baseline=120.0,
                bp_current=120.0 - sbp_drop
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Standard View")
        render_daily_stand_chart(st.session_state.test_event_logger)

    with col2:
        st.markdown("### Compact View")
        render_compact_daily_stand_chart(st.session_state.test_event_logger)

    st.markdown("### Mock Data Demo")
    render_daily_stand_chart(st.session_state.test_event_logger, show_mock=True)

    st.divider()
    st.markdown("### Activity Heatmap Demo")
    render_stand_activity_heatmap(st.session_state.test_event_logger, show_mock=True)

    st.divider()
    st.markdown("### Weekly Activity Summary Demo")
    render_weekly_activity_summary(st.session_state.test_event_logger, show_mock=True)

    st.divider()
    st.markdown("### Longitudinal Mobility Trend Demo")
    render_longitudinal_mobility_trend(st.session_state.test_event_logger, show_mock=True)
