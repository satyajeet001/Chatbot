const selectImage = document.querySelector('.select-img');
const inputFile = document.querySelector('#file');
const imgArea = document.querySelector('.img-area');

selectImage.addEventListener('click', function () {
    inputFile.click();
});

document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("video");
    const snapshotButton = document.getElementById("snapshot");
    const canvas = document.createElement("canvas"); // Create a new canvas element
    const context = canvas.getContext("2d");

    // Check if the browser supports getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                // Assign the stream to the video element
                video.srcObject = stream;
            })
            .catch((error) => {
                console.error("Error accessing camera:", error);
            });
    } else {
        console.error("getUserMedia is not supported by this browser");
    }

    snapshotButton.addEventListener("click", () => {
        // Set the canvas dimensions to match the video frame
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw the video frame on the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get the image data from the canvas as a data URL
        const imageDataURL = canvas.toDataURL("image/png");

        // Convert data URL to blob
        fetch(imageDataURL)
            .then(res => res.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append("image", blob);

                // Send the image data to the Flask server using FormData
                fetch('/upload_image', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
    });
});

inputFile.addEventListener('change', function () {
    const image = this.files[0];
    if (image.size < 10000000) {
        const reader = new FileReader();
        reader.onload = () => {
            const allImg = imgArea.querySelectorAll('img');
            allImg.forEach(item => item.remove());
            const imgUrl = reader.result;
            const img = document.createElement('img');
            img.src = imgUrl;
            imgArea.appendChild(img);
            imgArea.classList.add('active');
            imgArea.dataset.img = image.name;
        };
        reader.readAsDataURL(image);
    } else {
        alert("Image size more than 10MB");
    }
});
