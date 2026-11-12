
const camera = document.getElementById("camera");
const startRegistration = document.getElementById("startRegistration");
const startRecognition = document.getElementById("startRecognition");
const stopCamera = document.getElementById("stopCamera");
const saveRegistration = document.getElementById("saveRegistration");
const usernameInput = document.getElementById("username");
const statusText = document.getElementById("status");
const canvas = document.getElementById("canvas");

let cameraStream = null;
let detectionTimer = null;
let currentMode = null;
let isChecking = false;
let collectedEmbeddings = [];

async function startCamera(mode) {
    try {
        if (cameraStream) {
            stopCameraStream();
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        camera.srcObject = cameraStream;
        await camera.play();

        currentMode = mode;

        if (mode === "register") {
            collectedEmbeddings = [];
            saveRegistration.disabled = true;
            statusText.textContent =
                "Registration started. Look at the camera.";
        } else {
            statusText.textContent =
                "Face detection started. Look at the camera.";
        }

        stopCamera.disabled = false;
        startRegistration.disabled = true;
        startRecognition.disabled = true;

        // Check a webcam frame every 800 milliseconds
        detectionTimer = setInterval(checkFace, 800);

    } catch (error) {
        console.error(error);
        statusText.textContent =
            "Could not open camera. Check browser permissions.";
    }
}

async function checkFace() {
    if (!cameraStream || isChecking || camera.readyState < 2) {
        return;
    }

    isChecking = true;

    try {
        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;

        const context = canvas.getContext("2d");
        context.drawImage(camera, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, "image/jpeg", 0.85);
        });

        if (!blob) {
            throw new Error("Could not capture webcam image.");
        }

        const formData = new FormData();
        formData.append("frame", blob, "webcam.jpg");

        
        const endpoint = currentMode === "recognize"
            ? "/api/recognize"
            : "/api/face_embedding";

        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        const result = await response.json();


        if (!result.success) {
            statusText.textContent = result.message;
            return;
        }

        
        if (currentMode === "register") {
            if (collectedEmbeddings.length < 10) {
                collectedEmbeddings.push(result.embedding);
            }
        

            statusText.textContent =
            `Face detected! Samples: ${collectedEmbeddings.length}/10`;

            if (collectedEmbeddings.length >= 10) {
                saveRegistration.disabled = false;
                statusText.textContent =
                    "10 samples collected. You can save this person.";
            }
        } else {
            statusText.textContent = result.message;

            if (result.recognized) {
                statusText.textContent =
                    `Recognized: ${result.username} ` +
                    `(similarity: ${result.similarity})`;
            }
        }


    } catch (error) {
        console.error("Face detection error:", error);
        statusText.textContent =
            "Could not contact Flask. Check the server terminal.";
    } finally {
        isChecking = false;
    }
}

function stopCameraStream() {
    if (detectionTimer) {
        clearInterval(detectionTimer);
        detectionTimer = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    camera.srcObject = null;
    currentMode = null;
    isChecking = false;

    stopCamera.disabled = true;
    startRegistration.disabled = false;
    startRecognition.disabled = false;
}

startRegistration.addEventListener("click", () => {
    startCamera("register");
});

startRecognition.addEventListener("click", () => {
    startCamera("recognize");
});

stopCamera.addEventListener("click", () => {
    stopCameraStream();
    statusText.textContent = "Camera stopped.";
});

// Save the collected embeddings to Supabase through Flask
saveRegistration.addEventListener("click", async () => {
    const username = usernameInput.value.trim();

    if (!username) {
        statusText.textContent = "Please enter a username first.";
        return;
    }

    if (collectedEmbeddings.length < 10) {
        statusText.textContent = "Collect 10 samples first.";
        return;
    }

    saveRegistration.disabled = true;
    statusText.textContent = "Saving person...";

    try {
        const response = await fetch("/api/register", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username: username,
                embeddings: collectedEmbeddings
            })
        });

        const result = await response.json();
        statusText.textContent = result.message;

        if (result.success) {
            collectedEmbeddings = [];
            usernameInput.value = "";
            await loadPeople();
        }

    } catch (error) {
        console.error(error);
        statusText.textContent = "Could not save person.";
    } finally {
        saveRegistration.disabled =
            collectedEmbeddings.length < 10;
    }
});

async function loadPeople() {
    const peopleList = document.getElementById("peopleList");

    try {
        const response = await fetch("/api/people");
        const result = await response.json();

        peopleList.replaceChildren();

        if (!result.success) {
            peopleList.textContent = "Could not load people.";
            return;
        }

        result.people.forEach(person => {
            const item = document.createElement("li");
            item.textContent =
                `${person.username} — ${person.samples} samples`;
            peopleList.appendChild(item);
        });

    } catch (error) {
        console.error(error);
        peopleList.textContent = "Could not connect to Flask.";
    }
}

document.getElementById("refreshPeople")
    .addEventListener("click", loadPeople);

loadPeople();
