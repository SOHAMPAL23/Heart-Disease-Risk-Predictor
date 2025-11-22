import requests

try:
    # Try IPv4 address
    response = requests.get('http://127.0.0.1:3000/')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}...")
except Exception as e:
    print(f"Error with IPv4: {e}")
    
    try:
        # Try localhost
        response = requests.get('http://localhost:3000/')
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
    except Exception as e2:
        print(f"Error with localhost: {e2}")