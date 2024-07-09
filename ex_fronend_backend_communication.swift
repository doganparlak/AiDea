func sendDataToBackend() {
    // URL of your Flask endpoint
    let urlString = "http://yourflaskserver.com/process_data"
    
    // Data to send
    let dataToSend: [String: Any] = [
        "symbol": "AAPL",
        "data_length": 30,
        "forecast_days": 7,
        "model_type": "AR"
    ]
    
    // Convert data to JSON format
    guard let jsonData = try? JSONSerialization.data(withJSONObject: dataToSend) else {
        print("Error converting data to JSON")
        return
    }
    
    // Prepare URLRequest
    var request = URLRequest(url: URL(string: urlString)!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = jsonData
    
    // Perform the POST request
    URLSession.shared.dataTask(with: request) { (data, response, error) in
        // Check for errors
        if let error = error {
            print("Error sending data to server: \(error.localizedDescription)")
            return
        }
        
        // Check if response is received
        guard let httpResponse = response as? HTTPURLResponse else {
            print("No HTTP response received")
            return
        }
        
        // Check if response status code is OK (200)
        if httpResponse.statusCode == 200 {
            // Parse JSON response
            if let responseData = data {
                do {
                    // Convert JSON response to dictionary
                    if let json = try JSONSerialization.jsonObject(with: responseData, options: []) as? [String: Any] {
                        // Process the received data
                        if let forecastPrices = json["forecast_prices"] as? [Double] {
                            print("Received forecast prices: \(forecastPrices)")
                            // Update UI or perform further operations
                        }
                    }
                } catch {
                    print("Error parsing JSON response: \(error.localizedDescription)")
                }
            }
        } else {
            print("Unexpected response status code: \(httpResponse.statusCode)")
        }
    }.resume()
}
