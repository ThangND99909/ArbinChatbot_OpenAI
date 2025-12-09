import React, { useState } from 'react';
import Header from './components/Header';
import Chat from './components/Chat';
import InputArea from './components/InputArea';
import './styles.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (message) => {
    const userMessage = { text: message, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      const botMessage = { 
        text: data.answer, 
        sender: 'bot',
        sources: data.sources,
        intent: data.intent
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { 
        text: 'Sorry, there was an error processing your request.', 
        sender: 'bot' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />
      <div className="main-content">
        <Chat messages={messages} isLoading={isLoading} />
        <InputArea onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}

export default App;