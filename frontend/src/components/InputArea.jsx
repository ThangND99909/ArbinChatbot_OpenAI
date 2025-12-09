import React, { useState } from 'react';

function InputArea({ onSendMessage, isLoading }) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form className="input-area" onSubmit={handleSubmit}>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your question about Arbin..."
        disabled={isLoading}
      />
      <button type="submit" disabled={isLoading || !message.trim()}>
        {isLoading ? 'Sending...' : 'Send'}
      </button>
    </form>
  );
}

export default InputArea;