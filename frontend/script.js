// API endpoint - adjust this if your backend runs on a different port
const API_URL = 'http://localhost:8002';

// Store prescriptions in memory
let prescriptions = [];

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

// Prescription functionality
document.addEventListener('DOMContentLoaded', function() {
    // Ensure result section is hidden on page load
    const resultSection = document.getElementById('result-section');
    resultSection.classList.remove('show');
    
    // Add prescription button event
    document.getElementById('add-prescription-btn').addEventListener('click', addPrescription);
    
    // Save prescriptions button event
    document.getElementById('save-prescriptions-btn').addEventListener('click', savePrescriptions);
});

function addPrescription() {
    // Create prescription form
    const prescriptionDiv = document.createElement('div');
    prescriptionDiv.className = 'prescription-item';
    
    // Remove empty message if it exists
    const emptyMessage = document.getElementById('empty-prescription-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Generate unique ID for this prescription
    const prescriptionId = 'prescription_' + Date.now();
    
    prescriptionDiv.innerHTML = `
        <div class="prescription-header">
            <h4>Prescription #${prescriptions.length + 1}</h4>
            <button class="remove-prescription-btn" onclick="removePrescription('${prescriptionId}')">Remove</button>
        </div>
        <div class="prescription-form">
            <div class="form-row">
                <div class="form-group">
                    <label for="${prescriptionId}_medication">Medication:</label>
                    <input type="text" id="${prescriptionId}_medication" name="medication" placeholder="Enter medication name" required>
                </div>
                <div class="form-group">
                    <label for="${prescriptionId}_dosage">Dosage:</label>
                    <input type="text" id="${prescriptionId}_dosage" name="dosage" placeholder="e.g., 10mg" required>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="${prescriptionId}_frequency">Frequency:</label>
                    <select id="${prescriptionId}_frequency" name="frequency">
                        <option value="once_daily">Once Daily</option>
                        <option value="twice_daily">Twice Daily</option>
                        <option value="three_times_daily">Three Times Daily</option>
                        <option value="as_needed">As Needed</option>
                        <option value="before_meals">Before Meals</option>
                        <option value="after_meals">After Meals</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="${prescriptionId}_duration">Duration (days):</label>
                    <input type="number" id="${prescriptionId}_duration" name="duration" min="1" placeholder="Days">
                </div>
            </div>
            <div class="form-group">
                <label for="${prescriptionId}_instructions">Special Instructions:</label>
                <textarea id="${prescriptionId}_instructions" name="instructions" placeholder="Additional notes..."></textarea>
            </div>
        </div>
    `;
    
    // Add to prescription list
    document.getElementById('prescription-list').appendChild(prescriptionDiv);
    
    // Add to prescriptions array
    prescriptions.push({
        id: prescriptionId,
        element: prescriptionDiv
    });
}

function removePrescription(prescriptionId) {
    // Remove from DOM
    const prescriptionElement = document.querySelector(`.prescription-item [id^="${prescriptionId}"]`).closest('.prescription-item');
    if (prescriptionElement) {
        prescriptionElement.remove();
    }
    
    // Remove from array
    prescriptions = prescriptions.filter(p => p.id !== prescriptionId);
    
    // Update prescription numbers
    updatePrescriptionNumbers();
    
    // Show empty message if no prescriptions
    if (prescriptions.length === 0) {
        const emptyMessage = document.createElement('div');
        emptyMessage.id = 'empty-prescription-message';
        emptyMessage.className = 'empty-prescription-message';
        emptyMessage.textContent = 'No prescriptions added yet. Click "Add New Prescription" to get started.';
        document.getElementById('prescription-list').appendChild(emptyMessage);
    }
}

function updatePrescriptionNumbers() {
    const prescriptionItems = document.querySelectorAll('.prescription-item');
    prescriptionItems.forEach((item, index) => {
        const header = item.querySelector('.prescription-header h4');
        if (header) {
            header.textContent = `Prescription #${index + 1}`;
        }
    });
}

function savePrescriptions() {
    // Collect all prescription data
    const prescriptionData = [];
    
    prescriptions.forEach(prescription => {
        const medication = document.getElementById(`${prescription.id}_medication`).value;
        const dosage = document.getElementById(`${prescription.id}_dosage`).value;
        const frequency = document.getElementById(`${prescription.id}_frequency`).value;
        const duration = document.getElementById(`${prescription.id}_duration`).value;
        const instructions = document.getElementById(`${prescription.id}_instructions`).value;
        
        if (medication && dosage) { // Only add if medication and dosage are provided
            prescriptionData.push({
                id: prescription.id,
                medication,
                dosage,
                frequency,
                duration: duration || '',
                instructions: instructions || ''
            });
        }
    });
    
    if (prescriptionData.length === 0) {
        alert('No valid prescriptions to save!');
        return;
    }
    
    // Save to localStorage
    localStorage.setItem('heartDiseasePrescriptions', JSON.stringify(prescriptionData));
    
    // Show confirmation
    alert(`${prescriptionData.length} prescription(s) saved successfully!`);
    
    // Optionally send to backend if needed
    // sendPrescriptionsToBackend(prescriptionData);
}

function sendPrescriptionsToBackend(prescriptionData) {
    // This would be used if we had a backend endpoint for saving prescriptions
    /*
    fetch(`${API_URL}/prescriptions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            patientData: getFormData(),
            prescriptions: prescriptionData
        })
    })
    .then(response => response.json())
    .then(data => console.log('Prescriptions saved:', data))
    .catch(error => console.error('Error saving prescriptions:', error));
    */
}

function getFormData() {
    // Get current form data
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    const patientData = {};
    
    for (let [key, value] of formData.entries()) {
        if (key !== 'model') {
            patientData[key] = key === 'oldpeak' ? parseFloat(value) : parseInt(value);
        }
    }
    
    return patientData;
}

// Add input focus animations
document.querySelectorAll('input, select, textarea').forEach(element => {
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

// Expose functions to global scope for inline onclick handlers
window.removePrescription = removePrescription;