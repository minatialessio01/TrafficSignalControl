import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imposta lo stile di Seaborn
sns.set_theme(style="darkgrid")

# File path (formato WSL per essere eseguiti dentro docker)
colight_csv = "/workspace/results/colight_200m_config2/training_log.csv"
metastgat_csv = "/workspace/results/metastgat_20260903_123717/training_log.csv"

# Carica i DataFrame
df_colight = pd.read_csv(colight_csv)
df_metastgat = pd.read_csv(metastgat_csv)

# Rendi coerenti le colonne (aggiungi etichetta modello)
df_colight['Model'] = 'CoLight'
df_metastgat['Model'] = 'MetaSTGAT'

# MetaSTGAT ha probabilmente più episodi (es 100), CoLight ne ha 70.
# Tagliamo MetaSTGAT a 70 per avere un paragone alla pari sul grafico, se vogliamo.
# Oppure semplicemente li plottiamo entrambi così come sono
if 'episode' not in df_colight.columns and 'Episode' in df_colight.columns:
    df_colight.rename(columns={'Episode': 'episode'}, inplace=True)
if 'episode' not in df_metastgat.columns and 'Episode' in df_metastgat.columns:
    df_metastgat.rename(columns={'Episode': 'episode'}, inplace=True)

df_combined = pd.concat([df_colight, df_metastgat], ignore_index=True)

# Definisci le metriche da confrontare (assumendo le colonne standard di training_log.csv)
# Di solito le colonne sono "episode, travel_time, throughput, loss, reward" o simili
# Scopriamo i nomi delle colonne
print(f"Colonne CoLight: {list(df_colight.columns)}")
print(f"Colonne MetaSTGAT: {list(df_metastgat.columns)}")

# Se le colonne sono diverse, normalizziamole
def get_col_name(df, possible_names):
    for n in possible_names:
        if n in df.columns:
            return n
    return None

tt_col_colight = get_col_name(df_colight, ['travel_time', ' travel_time', 'travel time', 'avg_travel_time'])
tt_col_meta = get_col_name(df_metastgat, ['travel_time', ' travel_time', 'travel time', 'avg_travel_time'])

tp_col_colight = get_col_name(df_colight, ['throughput', ' throughput'])
tp_col_meta = get_col_name(df_metastgat, ['throughput', ' throughput'])

if tt_col_colight and tt_col_meta:
    df_colight['TravelTime'] = df_colight[tt_col_colight]
    df_metastgat['TravelTime'] = df_metastgat[tt_col_meta]

if tp_col_colight and tp_col_meta:
    df_colight['Throughput'] = df_colight[tp_col_colight]
    df_metastgat['Throughput'] = df_metastgat[tp_col_meta]

df_colight['TravelTime_MA'] = df_colight['TravelTime'].rolling(window=10, min_periods=1).mean()
df_metastgat['TravelTime_MA'] = df_metastgat['TravelTime'].rolling(window=10, min_periods=1).mean()
df_colight['Throughput_MA'] = df_colight['Throughput'].rolling(window=10, min_periods=1).mean()
df_metastgat['Throughput_MA'] = df_metastgat['Throughput'].rolling(window=10, min_periods=1).mean()

df_combined_normalized = pd.concat([df_colight, df_metastgat], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Travel Time
# Raw data with low alpha
sns.lineplot(data=df_combined_normalized, x='episode', y='TravelTime', hue='Model', ax=axes[0], alpha=0.3, linewidth=1)
# Moving Average with high alpha
sns.lineplot(data=df_combined_normalized, x='episode', y='TravelTime_MA', hue='Model', ax=axes[0], linewidth=2.5, legend=False)
axes[0].set_title('Travel Time over Episodes', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Avg Travel Time (s)', fontsize=12)

# Plot Throughput
# Raw data with low alpha
sns.lineplot(data=df_combined_normalized, x='episode', y='Throughput', hue='Model', ax=axes[1], alpha=0.3, linewidth=1)
# Moving Average with high alpha
sns.lineplot(data=df_combined_normalized, x='episode', y='Throughput_MA', hue='Model', ax=axes[1], linewidth=2.5, legend=False)
axes[1].set_title('Throughput over Episodes', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Vehicles Finished', fontsize=12)

plt.tight_layout()
output_path = "/workspace/results/colight_200m_config2/comparison_plot_ma.png"
plt.savefig(output_path, dpi=300)
print(f"Saved plot to {output_path}")

# Stampa statistiche veloci per report
print("\n--- STATISTICHE UTILI ---")
for model, df in zip(['CoLight', 'MetaSTGAT'], [df_colight, df_metastgat]):
    # consideriamo gli ultimi 10 episodi per la stabilità
    last_10 = df.tail(10)
    print(f"\n{model} (Ultimi 10 episodi):")
    print(f"  Travel Time Medio: {last_10['TravelTime'].mean():.2f}s")
    print(f"  Travel Time Minimo (Best): {df['TravelTime'].min():.2f}s")
    print(f"  Throughput Medio: {last_10['Throughput'].mean():.2f}")
    print(f"  Throughput Massimo (Best): {df['Throughput'].max():.2f}")
