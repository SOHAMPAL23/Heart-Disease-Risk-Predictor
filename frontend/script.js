// API endpoint - adjust this if your backend runs on a different port
const API_URL = 'http://localhost:8002';

// Form submission handler
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(this);
    const patientData = {};
    
    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        if (key !== 'model') {
            patientData[key] = key === 'oldpeak' ? parseFloat(value) : parseInt(value);
        }
    }
    
    try {
        // Show loading state
        const submitBtn = document.querySelector('.predict-btn');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="loading"></span> Predicting...';
        submitBtn.disabled = true;
        
        // Add animation class to form
        this.classList.add('predicting');
        
        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results with animations
        displayResults(result);
        
        // Restore button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
        // Remove animation class from form
        this.classList.remove('predicting');
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please check that the backend server is running.');
        
        // Restore button
        const submitBtn = document.querySelector('.predict-btn');
        submitBtn.innerHTML = 'Predict Risk';
        submitBtn.disabled = false;
        
        // Remove animation class from form
        this.classList.remove('predicting');
    }
});

function displayResults(result) {
    // Show result section with animation
    const resultSection = document.getElementById('result-section');
    resultSection.classList.add('show');
    
    // Update risk card with animation
    const riskCard = document.getElementById('risk-card');
    riskCard.className = 'risk-card'; // Reset classes
    riskCard.classList.add(result.risk_level === 'High Risk' ? 'high-risk' : 'low-risk');
    
    // Add slight delay for staggered animations
    setTimeout(() => {
        // Update risk indicator and level
        document.getElementById('risk-level').textContent = result.risk_level;
        
        // Update probability with percentage and progress bar
        const probabilityPercent = (result.probability * 100).toFixed(1);
        document.getElementById('probability').innerHTML = `Probability: <span>${probabilityPercent}%</span>`;
        
        // Update progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'probability-bar';
        progressBar.innerHTML = `<div class="probability-fill" style="width: ${probabilityPercent}%"></div>`;
        const probabilityElement = document.getElementById('probability');
        probabilityElement.parentNode.insertBefore(progressBar, probabilityElement.nextSibling);
        
        // Update clinical note
        document.getElementById('clinical-note').textContent = result.clinical_note;
        
        // Update model used
        document.getElementById('model-used').textContent = result.model_used;
        
        // Update contributing factors
        const factorsList = document.getElementById('factors-list');
        factorsList.innerHTML = '';
        
        result.contributing_factors.forEach(factor => {
            const li = document.createElement('li');
            if (factor.coefficient !== undefined) {
                // Logistic Regression - show coefficient
                li.innerHTML = `
                    <span>${factor.feature}</span>
                    <span>Coef: ${factor.coefficient.toFixed(3)}</span>
                `;
            } else if (factor.importance !== undefined) {
                // Random Forest - show importance
                li.innerHTML = `
                    <span>${factor.feature}</span>
                    <span>Imp: ${(factor.importance * 100).toFixed(1)}%</span>
                `;
            }
            factorsList.appendChild(li);
        });
    }, 100);
    
    // Scroll to results with smooth animation
    setTimeout(() => {
        resultSection.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }, 300);
}

// Add input focus animations
document.querySelectorAll('input, select').forEach(element => {
    element.addEventListener('focus', function() {
        this.parentElement.classList.add('focused');
    });
    
    element.addEventListener('blur', function() {
        this.parentElement.classList.remove('focused');
    });
});

// Add form reset functionality
document.getElementById('prediction-form').addEventListener('reset', function() {
    // Hide results section
    document.getElementById('result-section').classList.remove('show');
    
    // Reset all form fields
    this.reset();
});

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Ensure result section is hidden on page load
    const resultSection = document.getElementById('result-section');
    resultSection.classList.remove('show');
});