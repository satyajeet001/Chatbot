const chatBox = document.getElementById("chat-box");
const userMessageInput = document.getElementById("user-message");
const sendButton = document.getElementById("send-button");

function addMessage(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(`${sender}-message`);
    messageDiv.textContent = message;
    chatBox.appendChild(messageDiv);
    
}

addMessage("Hello, I'm your friendly bot! How can I help you?", "bot");

sendButton.addEventListener("click", () => {
    const userMessage = userMessageInput.value;
    if (userMessage.trim() !== "") {
        addMessage(userMessage, "user");
        userMessageInput.value = "";
        fetch("/get_response", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: `user_input=${encodeURIComponent(userMessage)}`,
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.response, "bot");
        });
    }
});
