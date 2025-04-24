document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearButton = document.getElementById('clear-button');
    const analyzeButton = document.getElementById('analyze-button');

    // Sample bot responses to randomly choose from
    const botResponses = [
        "I'm here to help! What would you like to know?",
        "That's an interesting question. Let me think about that.",
        "I understand your query. Here's what I can tell you...",
        "Thanks for sharing that with me. Would you like to know more?",
        "I'm not sure I fully understand. Could you rephrase that?",
        "Based on what you've told me, I'd recommend the following...",
        "That's a great point! I hadn't considered that perspective.",
        "I'm sorry, but I don't have enough information to answer that properly.",
        "Let me clarify a few things about your question...",
        "I've processed your request. Here's what I found."
    ];

    // Format time for timestamp
    function formatTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    // Function to create typing indicator
    function createTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot-message', 'typing-indicator');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');

        const icon = document.createElement('i');
        icon.classList.add('fas', 'fa-robot');
        avatar.appendChild(icon);

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        const dots = document.createElement('div');
        dots.classList.add('typing-dots');

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dots.appendChild(dot);
        }

        messageContent.appendChild(dots);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        return messageDiv;
    }

    // Function to add a message to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');

        const icon = document.createElement('i');
        icon.classList.add('fas');
        icon.classList.add(isUser ? 'fa-user' : 'fa-robot');
        avatar.appendChild(icon);

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        const paragraph = document.createElement('p');
        paragraph.textContent = message;
        messageContent.appendChild(paragraph);

        // Add timestamp
        const messageFooter = document.createElement('div');
        messageFooter.classList.add('message-footer');

        const timestamp = document.createElement('span');
        timestamp.classList.add('timestamp');
        timestamp.textContent = formatTime();
        messageFooter.appendChild(timestamp);

        messageContent.appendChild(messageFooter);

        if (isUser) {
            messageDiv.appendChild(messageContent);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
        }

        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to handle user input
    function handleUserInput() {
        const message = userInput.value.trim();
        if (message) {
            // Add user message
            addMessage(message, true);

            // Clear input field
            userInput.value = '';

            // Add typing indicator
            const typingIndicator = createTypingIndicator();
            chatMessages.appendChild(typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Simulate bot "typing" with a delay
            setTimeout(() => {
                // Remove typing indicator
                chatMessages.removeChild(typingIndicator);

                // Get random bot response
                const randomIndex = Math.floor(Math.random() * botResponses.length);
                const botResponse = botResponses[randomIndex];

                // Add bot response
                addMessage(botResponse);
            }, 1500);
        }
    }

    // Function to clear chat
    function clearChat() {
        // Keep only the first message (welcome message)
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }

        // Update the timestamp of the first message
        const firstMessageFooter = chatMessages.querySelector('.message-footer');
        if (firstMessageFooter) {
            const timestamp = firstMessageFooter.querySelector('.timestamp');
            if (timestamp) {
                timestamp.textContent = formatTime();
            }
        }
    }

    // Function to download chat history as text
    function downloadChatHistory() {
        // Collect all user messages
        const userMessages = Array.from(document.querySelectorAll('.user-message .message-content p'))
            .map(p => p.textContent)
            .join('\n\n');
        
        if (!userMessages) {
            addMessage("No messages to analyze. Please chat with me first!", false);
            return false;
        }
        
        // Create a blob with the text content
        const blob = new Blob([userMessages], { type: 'text/plain' });
        
        // Send the text for MBTI analysis
        analyzeMBTI(userMessages);
        
        return true;
    }
    
    // Function to send text to backend for MBTI analysis
    async function analyzeMBTI(text) {
        try {
            // Show typing indicator while waiting for response
            const typingIndicator = createTypingIndicator();
            chatMessages.appendChild(typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // In a real implementation, you would send this to your backend
            // For now, we'll simulate a response
            const response = await fetch('/analyze-mbti', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            if (!response.ok) {
                throw new Error('Failed to analyze personality');
            }
            
            const result = await response.json();
            
            // Remove typing indicator
            chatMessages.removeChild(typingIndicator);
            
            // Display the MBTI result
            const mbtiMessage = `Based on our analysis, your personality type is: ${result.mbti_type} (Confidence: ${(result.confidence * 100).toFixed(2)}%)
            
Here's what that means:
${result.explanation}`;
            
            addMessage(mbtiMessage, false);
            
        } catch (error) {
            console.error('Error analyzing MBTI:', error);
            
            // Remove typing indicator if it exists
            const indicator = document.querySelector('.typing-indicator');
            if (indicator) {
                chatMessages.removeChild(indicator);
            }
            
            // Show error message
            addMessage("Sorry, I couldn't analyze your personality at this time. Please try again later.", false);
        }
    }

    // Send button click event
    sendButton.addEventListener('click', handleUserInput);

    // Analyze button click event
    analyzeButton.addEventListener('click', function() {
        downloadChatHistory();
    });

    // Clear chat button click event
    clearButton.addEventListener('click', clearChat);

    // Enter key press event
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default to avoid newline
            handleUserInput();
        }
    });

    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        // Reset height to auto to get the correct scrollHeight
        this.style.height = 'auto';

        // Calculate new height (min is original height, max is 120px)
        const newHeight = Math.min(this.scrollHeight, 120);

        // Set the new height
        this.style.height = newHeight + 'px';
    });

    // Focus on input when page loads
    userInput.focus();
});