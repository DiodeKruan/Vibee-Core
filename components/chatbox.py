"""Floating chatbox component rendered with HTML/CSS."""

import streamlit as st


def render_chatbox():
    """Render a styled bottom-right chatbox that echoes user input."""
    component_id = "vibee-chatbox"

    st.markdown(
        f"""
        <style>
        .chatbox-wrapper {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 480px;
            max-width: 92vw;
            z-index: 60;
            font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
        }}

        .chatbox {{
            background: linear-gradient(145deg, rgba(10, 10, 15, 0.95), rgba(15, 15, 22, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55);
            overflow: hidden;
            backdrop-filter: blur(16px);
        }}

        .chatbox-header {{
            padding: 12px 14px;
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.18), rgba(245, 158, 11, 0.12));
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .chatbox-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #f97316;
            box-shadow: 0 0 0 6px rgba(249, 115, 22, 0.12);
        }}

        .chatbox-title {{
            color: #e2e8f0;
            font-weight: 700;
            letter-spacing: -0.2px;
            font-size: 13px;
        }}

        .chatbox-subtitle {{
            color: #94a3b8;
            font-size: 11px;
            margin-left: auto;
        }}

        .chatbox-messages {{
            max-height: 600px;
            min-height: 300px;
            overflow-y: auto;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            background: radial-gradient(circle at 20% 20%, rgba(249, 115, 22, 0.06), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.06), transparent 18%);
        }}

        .chatbox-empty {{
            color: #94a3b8;
            font-size: 12px;
            text-align: center;
            padding: 12px;
            border: 1px dashed rgba(255, 255, 255, 0.08);
            border-radius: 10px;
        }}

        .chatbox-message {{
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }}

        .chatbox-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 8px;
            background: linear-gradient(135deg, #f97316, #f59e0b);
            color: #0f172a;
            font-weight: 700;
            display: grid;
            place-items: center;
            font-size: 12px;
            box-shadow: 0 8px 18px rgba(249, 115, 22, 0.28);
        }}

        .chatbox-bubble {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 12px;
            line-height: 1.5;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }}

        .chatbox-input {{
            display: flex;
            gap: 8px;
            padding: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(6, 8, 15, 0.8);
        }}

        .chatbox-input input {{
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
            padding: 10px 12px;
            border-radius: 10px;
            font-size: 12px;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}

        .chatbox-input input:focus {{
            border-color: rgba(249, 115, 22, 0.6);
            box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.18);
        }}

        .chatbox-input button {{
            background: linear-gradient(135deg, #f97316, #f59e0b);
            border: none;
            color: #0f172a;
            font-weight: 700;
            padding: 10px 14px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 12px;
            box-shadow: 0 12px 28px rgba(249, 115, 22, 0.35);
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        }}

        .chatbox-input button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 14px 32px rgba(249, 115, 22, 0.45);
        }}

        .chatbox-input button:active {{
            transform: translateY(0);
            box-shadow: 0 8px 18px rgba(249, 115, 22, 0.32);
        }}

        @media (max-width: 640px) {{
            .chatbox-wrapper {{
                right: 12px;
                left: 12px;
                width: auto;
            }}
        }}
        </style>

        <div class="chatbox-wrapper" data-chatbox-id="{component_id}">
            <div class="chatbox">
                <div class="chatbox-header">
                    <div class="chatbox-dot"></div>
                    <div class="chatbox-title">Live Chat</div>
                    <div class="chatbox-subtitle">Echo mode</div>
                </div>
                <div class="chatbox-messages" id="{component_id}-messages">
                    <div class="chatbox-empty">Ask anything — I will echo it here.</div>
                </div>
                <form class="chatbox-input" id="{component_id}-form">
                    <input id="{component_id}-input" type="text" placeholder="Type a message and press Enter..." autocomplete="off" />
                    <button type="submit">Send</button>
                </form>
            </div>
        </div>

        <script>
        (function() {{
            const root = document.querySelector('[data-chatbox-id="{component_id}"]');
            if (!root || root.dataset.ready === "1") return;
            root.dataset.ready = "1";

            const form = document.getElementById("{component_id}-form");
            const input = document.getElementById("{component_id}-input");
            const messages = document.getElementById("{component_id}-messages");
            const emptyState = messages.querySelector('.chatbox-empty');

            const escapeHtml = (unsafe) => unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");

            const appendMessage = (text) => {{
                if (!text) return;
                if (emptyState) emptyState.remove();

                const wrap = document.createElement('div');
                wrap.className = 'chatbox-message';

                const avatar = document.createElement('div');
                avatar.className = 'chatbox-avatar';
                avatar.textContent = 'You';

                const bubble = document.createElement('div');
                bubble.className = 'chatbox-bubble';
                bubble.innerHTML = escapeHtml(text);

                wrap.appendChild(avatar);
                wrap.appendChild(bubble);
                messages.appendChild(wrap);
                messages.scrollTop = messages.scrollHeight;
            }};

            form.addEventListener('submit', (e) => {{
                e.preventDefault();
                const text = input.value.trim();
                if (!text) return;
                appendMessage(text);
                input.value = '';
                input.focus();
            }});
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

