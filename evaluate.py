import pandas as pd
import argparse
import sys

def compute_monthly_climate_futures_correlations(df):

    # Dynamic detection
    climate_cols = [c for c in df.columns if c.startswith("climate_risk_")]
    futures_cols = [c for c in df.columns if c.startswith("futures_")]

    # Remove future data
    max_valid_date = df["date_on"].max()
    df = df[df["date_on"] <= max_valid_date]

    results = []

    # Loop by commodity + month
    for comm in df["crop_name"].unique():
        df_comm = df[df["crop_name"] == comm]

        for country in sorted(df_comm["country_name"].unique()):
            df_country = df_comm[df_comm["country_name"] == country]

            for month in sorted(df_country["date_on_month"].unique()):
                df_month = df_country[df_country["date_on_month"] == month]

                for clim in climate_cols:
                    for fut in futures_cols:

                        if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                            corr = df_month[[clim, fut]].corr().iloc[0, 1]
                        else:
                            corr = None

                        results.append({
                            "crop_name": comm,
                            "country_name": country,
                            "month": month,
                            "climate_variable": clim,
                            "futures_variable": fut,
                            "correlation": corr
                        })

    results_df = pd.DataFrame(results)
    #  round correlation to 5 decimal places
    results_df['correlation'] = results_df['correlation'].round(5)
    return results_df

def calculate_cfcs_score(correlations_df):
    """
    Calculate the Climate-Futures Correlation Score (CFCS) for leaderboard ranking.
    
    CFCS = (0.5 × Avg_Sig_Corr_Score) + (0.3 × Max_Corr_Score) + (0.2 × Sig_Count_Score)
    
    Focus on significant correlations (≥ |0.5|) only for average calculation.
    """
    # Remove null correlations
    valid_corrs = correlations_df["correlation"].dropna()
    
    if len(valid_corrs) == 0:
        return {'cfcs_score': 0.0, 'error': 'No valid correlations'}
    
    # Calculate base metrics
    abs_corrs = valid_corrs.abs()
    max_abs_corr = abs_corrs.max()
    significant_mask = abs_corrs >= 0.5
    significant_corrs = abs_corrs[significant_mask]
    significant_count = len(significant_corrs)
    total_count = len(valid_corrs)
    
    # Calculate component scores - ONLY average significant correlations
    if significant_count > 0:
        avg_sig_corr = significant_corrs.mean()
        avg_sig_score = min(100, avg_sig_corr * 100)  # Cap at 100 when avg sig reaches 1.0
    else:
        avg_sig_corr = 0.0
        avg_sig_score = 0.0
    
    max_corr_score = min(100, max_abs_corr * 100)  # Cap at 100 when max reaches 1.0
    sig_count_score = (significant_count / total_count) * 100  # Percentage
    
    # Composite score: Focus more on quality of significant correlations
    cfcs = (0.5 * avg_sig_score) + (0.3 * max_corr_score) + (0.2 * sig_count_score)
    
    return {
        'cfcs_score': round(cfcs, 2),
        'avg_significant_correlation': round(avg_sig_corr, 4),
        'max_abs_correlation': round(max_abs_corr, 4),
        'significant_correlations_pct': round(sig_count_score, 2),
        'avg_sig_score': round(avg_sig_score, 2),
        'max_corr_score': round(max_corr_score, 2),
        'sig_count_score': round(sig_count_score, 2),
        'total_correlations': total_count,
        'significant_correlations': significant_count
    }

def print_cfcs_results(score_results):
    """Print the CFCS score results in a formatted manner."""
    print("=== CLIMATE-FUTURES CORRELATION SCORE (CFCS) ===")
    print(f"Final CFCS Score: {score_results['cfcs_score']}")
    print()
    print("Component Breakdown:")
    print(f"  Average Significant |Correlation|: {score_results['avg_significant_correlation']:.4f} → Score: {score_results['avg_sig_score']}")
    print(f"  Maximum |Correlation|: {score_results['max_abs_correlation']:.4f} → Score: {score_results['max_corr_score']}")
    print(f"  Significant Correlations: {score_results['significant_correlations']}/{score_results['total_correlations']} ({score_results['significant_correlations_pct']:.1f}%) → Score: {score_results['sig_count_score']}")
    print()
    print("Score Calculation:")
    print(f"  CFCS = (0.5 × {score_results['avg_sig_score']}) + (0.3 × {score_results['max_corr_score']}) + (0.2 × {score_results['sig_count_score']})")
    print(f"  CFCS = {0.5 * score_results['avg_sig_score']:.1f} + {0.3 * score_results['max_corr_score']:.1f} + {0.2 * score_results['sig_count_score']:.1f} = {score_results['cfcs_score']}")
    print()
    print("Key Insight: This metric focuses on the QUALITY of significant correlations rather than being diluted by weak signals.")

def evaluate_submission(csv_file):
    """Evaluate a submission CSV file and return correlation results."""
    # Load the CSV file
    merged_daily_df_copy = pd.read_csv(csv_file)
    
    # Set crop name
    merged_daily_df_copy['crop_name'] = 'Corn: Commodity Tracked'
    
    # Compute monthly correlations
    monthly_corr_df = compute_monthly_climate_futures_correlations(merged_daily_df_copy)
    
    # Calculate the CFCS score
    score_results = calculate_cfcs_score(monthly_corr_df)
    
    # Print results
    print_cfcs_results(score_results)
    
    # Get the significant correlations greater than 0.5 or less than -0.5
    significant_monthly_correlations = monthly_corr_df[
        (monthly_corr_df["correlation"] >= 0.5) | (monthly_corr_df["correlation"] <= -0.5)
    ]
    
    # Sort by correlation
    significant_monthly_correlations = significant_monthly_correlations.sort_values(by='correlation')
    
    print()
    print("=== TOP 10 SIGNIFICANT CORRELATIONS ===")
    print(significant_monthly_correlations.head(10).to_string(index=False))
    
    return score_results, significant_monthly_correlations

def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate corn climate risk futures submission CSV file'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the submission CSV file to evaluate'
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_submission(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()