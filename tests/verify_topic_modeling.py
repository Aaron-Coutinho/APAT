import pandas as pd
import sys
import os

# Add the project root to sys.path to allow imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
try:
    print(f"Runtime JAX version: {jax.__version__}")
    print(f"JAX tree_util has register_pytree_node: {'register_pytree_node' in dir(jax.tree_util)}")
except Exception as e:
    print(f"Error checking JAX: {e}")

from backend.topic_modeling import PatentTopicModeler

def test_topic_modeling_with_real_data():
    print("Testing Topic Modeling with real patent data...")
    
    # Load patents
    try:
        patents_df = pd.read_csv("data/patents_clean.csv")
    except FileNotFoundError:
        print("Data file not found, creating larger mock set...")
        # Create at least 15-20 documents for BERTopic to have something to work with
        mock_records = []
        techs = ["AI", "Blockchain", "Quantum", "Energy", "IoT"]
        for i in range(25):
            tech = techs[i % len(techs)]
            mock_records.append({
                "title": f"New discovery in {tech} version {i}",
                "abstract": f"This patent discusses {tech} improvements and its impact on the industry. It covers various aspects of {tech} research."
            })
        patents_df = pd.DataFrame(mock_records)

    modeler = PatentTopicModeler()
    
    # We'll use a subset if it's too large, but for testing, let's use the whole thing
    results_df = modeler.extract_topics_from_patents(patents_df)
    
    print("\nDiscovered Topics Summary:")
    print(modeler.get_topic_info())
    
    print("\nSample Mappings (First 10):")
    print(results_df[['title', 'topic_label']].head(10))
    
    assert 'topic_label' in results_df.columns
    assert not results_df.empty
    print("\nTopic Modeling verification passed!")

if __name__ == "__main__":
    try:
        test_topic_modeling_with_real_data()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
