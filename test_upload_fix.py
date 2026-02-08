import requests
import pandas as pd

# Create test file
test_data = {
    'feedback': [
        "This is the best policy ever! It will transform our community for the better.",
        "Terrible decision that will destroy small businesses and hurt families.",
        "Reasonable approach but needs more detailed implementation guidelines.",
        "Outstanding work by our representatives! This addresses all key concerns.",
        "Complete disaster - ignores expert advice and scientific evidence."
    ],
    'policy_type': ['Education', 'Economic', 'Healthcare', 'Environment', 'Education'],
    'rating': [5, 1, 3, 5, 1]
}

df = pd.DataFrame(test_data)
df.to_csv('test_upload.csv', index=False)
print("✅ Test file 'test_upload.csv' created!")