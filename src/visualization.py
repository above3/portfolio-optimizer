"""Plotly figures for the portfolio optimizer.

The main entry point is :func:`efficient_frontier_figure`, which assembles the
classic Markowitz chart: a cloud of random portfolios (the feasible set), the
efficient-frontier curve, the risk-free rate on the Y-axis, the Capital Market
Line tangent to the frontier, and markers for the MVP, the tangency portfolio
and the user's current portfolio.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .optimizer import FrontierResult, PortfolioPoint
from .portfolio import CurrentPortfolio


def _weights_hover(weights: dict[str, float], top: int = 8) -> str:
    """Render the largest weights as an HTML hover snippet."""
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    items = [(t, w) for t, w in items if abs(w) > 1e-4][:top]
    return "<br>".join(f"  {t}: {w * 100:.1f}%" for t, w in items)


def _point_hover(point: PortfolioPoint) -> str:
    return (
        f"<b>{point.name}</b><br>"
        f"Return: {point.expected_return * 100:.2f}%<br>"
        f"Volatility: {point.volatility * 100:.2f}%<br>"
        f"Sharpe: {point.sharpe:.2f}<br>"
        f"<br>Weights:<br>{_weights_hover(point.weights)}"
        "<extra></extra>"
    )


def efficient_frontier_figure(
    result: FrontierResult,
    current: CurrentPortfolio | None = None,
) -> go.Figure:
    """Build the interactive Markowitz / efficient-frontier figure."""
    fig = go.Figure()

    # ---- 1. Feasible set: cloud of random portfolios, coloured by Sharpe ----
    fig.add_trace(
        go.Scattergl(
            x=result.random_vol * 100,
            y=result.random_ret * 100,
            mode="markers",
            name="Simulated portfolios",
            marker=dict(
                size=4,
                color=result.random_sharpe,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe", x=1.02),
                opacity=0.55,
            ),
            hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # ---- 2. Efficient frontier curve ----
    fig.add_trace(
        go.Scatter(
            x=result.frontier_vol * 100,
            y=result.frontier_ret * 100,
            mode="lines",
            name="Efficient frontier",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # ---- 3. Capital Market Line (Rf -> tangency, extended) ----
    rf_pct = result.risk_free_rate * 100
    if result.tangency is not None:
        t = result.tangency
        # Slope of the CML is the tangency Sharpe ratio.
        x_max = float(
            max(
                result.random_vol.max() if result.random_vol.size else 0.0,
                t.volatility,
            )
        ) * 100 * 1.05
        slope = (t.expected_return * 100 - rf_pct) / (t.volatility * 100)
        cml_x = np.array([0.0, x_max])
        cml_y = rf_pct + slope * cml_x
        fig.add_trace(
            go.Scatter(
                x=cml_x,
                y=cml_y,
                mode="lines",
                name="Capital Market Line",
                line=dict(color="#2ca02c", width=2, dash="dash"),
                hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            )
        )

    # ---- 4. Risk-free rate marker on the Y-axis (x = 0) ----
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[rf_pct],
            mode="markers+text",
            name="Risk-free rate",
            text=[f"Rf = {rf_pct:.2f}%"],
            textposition="middle right",
            marker=dict(symbol="circle", size=10, color="#2ca02c"),
            hovertemplate=f"Risk-free rate: {rf_pct:.2f}%<extra></extra>",
        )
    )

    # ---- 5. Minimum-variance portfolio ----
    mvp = result.mvp
    fig.add_trace(
        go.Scatter(
            x=[mvp.volatility * 100],
            y=[mvp.expected_return * 100],
            mode="markers",
            name="Minimum variance",
            marker=dict(symbol="star", size=18, color="#ff7f0e",
                        line=dict(width=1, color="black")),
            hovertemplate=_point_hover(mvp),
        )
    )

    # ---- 6. Tangency / max-Sharpe portfolio ----
    if result.tangency is not None:
        t = result.tangency
        fig.add_trace(
            go.Scatter(
                x=[t.volatility * 100],
                y=[t.expected_return * 100],
                mode="markers",
                name="Max Sharpe (Tangency)",
                marker=dict(symbol="star", size=20, color="#d62728",
                            line=dict(width=1, color="black")),
                hovertemplate=_point_hover(t),
            )
        )

    # ---- 7. Market benchmark reference point ----
    if result.benchmark is not None:
        b = result.benchmark
        fig.add_trace(
            go.Scatter(
                x=[b.volatility * 100],
                y=[b.expected_return * 100],
                mode="markers",
                name=b.name,
                marker=dict(symbol="star-diamond", size=18, color="#FFD700",
                            line=dict(width=1.5, color="black")),
                hovertemplate=(
                    f"<b>{b.name}</b><br>"
                    f"Return: {b.expected_return * 100:.2f}%<br>"
                    f"Volatility: {b.volatility * 100:.2f}%<br>"
                    f"Sharpe: {b.sharpe:.2f}<extra></extra>"
                ),
            )
        )

    # ---- 8. Current portfolio ----
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[current.volatility * 100],
                y=[current.expected_return * 100],
                mode="markers",
                name="Current portfolio",
                marker=dict(symbol="diamond", size=16, color="#9467bd",
                            line=dict(width=1.5, color="black")),
                hovertemplate=(
                    "<b>Current portfolio</b><br>"
                    f"Return: {current.expected_return * 100:.2f}%<br>"
                    f"Volatility: {current.volatility * 100:.2f}%<br>"
                    f"Sharpe: {current.sharpe:.2f}<br>"
                    f"<br>Weights:<br>{_weights_hover(current.weights)}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Markowitz Efficient Frontier & Capital Market Line",
        xaxis_title="Annualised volatility (standard deviation) %",
        yaxis_title="Annualised expected return %",
        template="plotly_white",
        height=620,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangemode="tozero")
    return fig


def correlation_heatmap(corr) -> go.Figure:
    """Heatmap of the asset correlation matrix (diversification view)."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
            hovertemplate="%{y} vs %{x}<br>ρ = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Asset correlation matrix (ρ) — red = diversifying, blue = correlated",
        template="plotly_white",
        height=460,
        yaxis=dict(autorange="reversed"),
    )
    return fig
