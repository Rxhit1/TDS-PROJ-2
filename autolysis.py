import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_data(file_path):
    """
    Load a CSV file and return a Pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, encoding="utf-8")
        print(f"Data loaded successfully from {file_path}")
        return data
    except UnicodeDecodeError:
        print("Error: File encoding not supported. Trying 'ISO-8859-1'.")
        try:
            data = pd.read_csv(file_path, encoding="ISO-8859-1")
            print(f"Data loaded successfully from {file_path} with 'ISO-8859-1' encoding.")
            return data
        except Exception as e:
            print(f"Error loading file: {e}")
            exit()
    except FileNotFoundError:
        print("Error: File not found.")
        exit()
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
        exit()
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def analyze_data(data):
    """
    Perform basic data analysis and print summary.
    """
    print("\n--- Dataset Overview ---\n")
    print(data.info())
    print("\n--- Summary Statistics ---\n")
    print(data.describe())
    print("\n--- Missing Values ---\n")
    print(data.isnull().sum())

def visualize_data(data, output_dir):
    """
    Create visualizations and save them as PNG files.
    """
    try:
        sns.set(style="whitegrid")

        # Correlation heatmap (numeric columns only)
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if not numeric_data.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            print(f"Correlation heatmap saved to {heatmap_path}")
            plt.close()

        # Pairplot
        if not numeric_data.empty:
            pairplot_path = os.path.join(output_dir, "pairplot.png")
            sns.pairplot(numeric_data)
            plt.savefig(pairplot_path)
            print(f"Pairplot saved to {pairplot_path}")
            plt.close()

    except Exception as e:
        print(f"Error creating visualizations: {e}")

def generate_markdown_report(data, output_dir):
    """
    Generate a Markdown report summarizing the analysis.
    """
    try:
        report_path = os.path.join(output_dir, "README.md")
        with open(report_path, "w") as f:
            f.write("# Data Analysis Report\n\n")
            f.write("## Dataset Overview\n\n")
            f.write(f"Shape of the dataset: {data.shape}\n\n")
            f.write("### Missing Values\n\n")
            f.write(data.isnull().sum().to_string())
            f.write("\n\n")
            f.write("### Summary Statistics\n\n")
            f.write(data.describe().to_string())
            f.write("\n")
        print(f"Markdown report saved to {report_path}")
    except Exception as e:
        print(f"Error generating Markdown report: {e}")

def main():
    # Ensure the working directory is correct
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load the AI Proxy token from environment variables
    aip_token = os.getenv("AIPROXY_TOKEN")
    if not aip_token:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        exit()

    # Input datasets
    datasets = {
        "goodreads": "C:/Users/Rohit/goodreads.csv",
        "happiness": "C:/Users/Rohit/happiness.csv",
        "media": "C:/Users/Rohit/media.csv"
    }

    # Output directory for results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset
    for name, file_path in datasets.items():
        print(f"\nProcessing dataset: {name}")
        data = load_data(file_path)
        dataset_dir = os.path.join(output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        analyze_data(data)
        visualize_data(data, dataset_dir)
        generate_markdown_report(data, dataset_dir)

if __name__ == "__main__":
    main()

