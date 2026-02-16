# === eda.py (ACTUALIZADO) ===
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

sns.set(style="darkgrid")

def show_age_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['age'], bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def show_avg_value_by_position(df):
    avg_value = df.groupby('position')['value'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    colors = sns.color_palette("Greens", len(avg_value))
    bars = ax.bar(avg_value.index, avg_value.values, color=colors)
    for bar, val in zip(bars, avg_value.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val/1e6:.1f}M',
                ha='center', va='bottom')
    ax.set_title("Average Value by Position #1")
    ax.set_ylabel("Average Value ($)")
    ax.set_xlabel("Position")
    st.pyplot(fig)


def show_value_by_league(df):
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    sns.stripplot(data=df, x='league', y='value', ax=axs[0], palette='plasma', size=3)
    sns.boxplot(data=df, x='league', y='value', ax=axs[1], palette='cubehelix')
    axs[0].set_title("Value ($) by League")
    axs[1].set_ylabel("Value ($)")
    axs[1].set_xlabel("League")
    st.pyplot(fig)

def show_players_by_foot(df):
    foot_counts = df['foot'].value_counts()
    fig, ax = plt.subplots()
    colors = sns.color_palette("Purples", len(foot_counts))
    bars = ax.bar(foot_counts.index, foot_counts.values, color=colors)
    for bar, val in zip(bars, foot_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}', ha='center', va='bottom')
    ax.set_title("Nº of Players by Foot")
    ax.set_ylabel("No of Players")
    ax.set_xlabel("Foot")
    st.pyplot(fig)

def show_value_distribution_by_foot(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=df, x='foot', y='value', palette='Purples', ax=ax)
    ax.set_title("Value ($) Distribution by Foot")
    ax.set_ylabel("Value ($)")
    st.pyplot(fig)


def show_athletes_per_nation(df):
    counts = df['nationality'].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='summer')
    ax.set_title("Nº of Athletes per Nation")
    ax.set_ylabel("No of Athletes")
    ax.set_xlabel("Country")
    st.pyplot(fig)

def show_pairplot_selected(df, columns):
    fig = sns.pairplot(df[columns])
    st.pyplot(fig)


