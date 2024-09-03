document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');
    const loadingAnimation = document.getElementById('loading-animation');
    
    // Display the selected file name
    fileInput.addEventListener('change', function () {
        fileName.textContent = this.files[0].name;
    });

    // Show loading animation on form submit
    uploadForm.addEventListener('submit', function (event) {
        loadingAnimation.style.display = 'block';
    });
});
