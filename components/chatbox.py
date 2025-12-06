"""Floating chatbox component rendered with streamlit-float."""

from html import escape

import streamlit as st

try:
    from streamlit_float import float_init
except ImportError:
    float_init = None


def _append_message(role: str, content: str) -> None:
    """Store a chat message in session state."""
    st.session_state.chatbox_history.append({"role": role, "content": content})


def render_chatbox() -> None:
    """Render a styled bottom-right chatbox using streamlit-float."""
    if float_init:
        float_init()

    if "chatbox_history" not in st.session_state:
        st.session_state.chatbox_history = []

    component_id = "vibee-chatbox"
    
    chat_container = st.container(key="chatbox-container")

    float_css = (
        "position: fixed;bottom: 24px;right: 24px;"
        "width: min(480px, calc(100vw - 32px));max-width: 92vw;z-index: 60;"
    )
    if hasattr(chat_container, "float"):
        chat_container.float(float_css)
    elif float_init:
        st.info("streamlit-float installed but float() not available.", icon="ℹ️")

    messages_html = ""
    history = st.session_state.chatbox_history[-100:]
    if not history:
        messages_html = '<div class="chatbox-empty">Ask anything — I will echo it here.</div>'
    else:
        parts = []
        for message in history:
            role = message.get("role", "user")
            role_class = "user" if role == "user" else "assistant"
            content = escape(message.get("content", ""))
            avatar = ""
            if role == "user":
                avatar = '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M8.4 13.8C8.4 13.8 9.75 15.6 12 15.6C14.25 15.6 15.6 13.8 15.6 13.8M14.7 9.3H14.709M9.3 9.3H9.309M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12ZM15.15 9.3C15.15 9.54853 14.9485 9.75 14.7 9.75C14.4515 9.75 14.25 9.54853 14.25 9.3C14.25 9.05147 14.4515 8.85 14.7 8.85C14.9485 8.85 15.15 9.05147 15.15 9.3ZM9.75 9.3C9.75 9.54853 9.54853 9.75 9.3 9.75C9.05147 9.75 8.85 9.54853 8.85 9.3C8.85 9.05147 9.05147 8.85 9.3 8.85C9.54853 8.85 9.75 9.05147 9.75 9.3Z" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path> </g></svg>'
            elif role == "assistant":
                avatar = '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M9 10H9.009M15 10H15.009M5 10C5 6.13401 8.13401 3 12 3C15.866 3 19 6.13401 19 10V21L18.1828 20.4552C17.4052 19.9368 17.0165 19.6777 16.605 19.6045C16.2421 19.54 15.8685 19.577 15.5253 19.7114C15.1362 19.8638 14.8058 20.1942 14.145 20.855V20.855C14.0649 20.9351 13.9351 20.935 13.8549 20.855C13.4039 20.4052 13.1548 20.1686 12.888 20.0364C12.3285 19.7591 11.6715 19.7591 11.112 20.0364C10.8452 20.1686 10.5961 20.4052 10.1451 20.855C10.0649 20.935 9.93508 20.9351 9.855 20.855V20.855C9.19423 20.1942 8.86384 19.8638 8.47469 19.7114C8.13152 19.577 7.75788 19.54 7.39501 19.6045C6.98352 19.6777 6.59475 19.9368 5.81722 20.4552L5 21V10Z" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path> </g></svg>'
            parts.append(
                f"""
<div class="chatbox-message {role_class}">
  <div class="chatbox-avatar {role_class}">{avatar}</div>
  <div class="chatbox-bubble {role_class}">{content}</div>
</div>
"""
            )
        messages_html = "\n".join(parts)

    chat_container.markdown(
        f"""
        <style>
        .chatbox-wrapper {{
            width: 100%;
            max-width: 520px;
            font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
        }}
        .chatbox {{
            background: linear-gradient(145deg, rgba(9, 9, 14, 0.95), rgba(16, 16, 24, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-bottom: none !important;
            border-radius: 14px 14px 0 0;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.55);
            overflow: hidden;
            backdrop-filter: blur(18px);
        }}
        .chatbox-header {{
            padding: 12px 14px;
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(245, 158, 11, 0.1));
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
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
            max-height: 540px;
            min-height: 320px;
            overflow-y: auto;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            background: radial-gradient(circle at 18% 18%, rgba(249, 115, 22, 0.06), transparent 26%),
                        radial-gradient(circle at 78% 6%, rgba(59, 130, 246, 0.05), transparent 18%);
        }}
        .chatbox-empty {{
            color: #94a3b8;
            font-size: 12px;
            text-align: center;
            padding: 12px;
            border: 1px dashed rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}
        .chatbox-message {{
            display: flex;
            gap: 10px;
            align-items: flex-start;
            width: 100%;
        }}
        .chatbox-message.user {{
            flex-direction: row-reverse;
        }}
        .chatbox-avatar {{
            width: 28px;
            height: 28px;
            padding: 3px;
            border-radius: 100%;
            display: grid;
            place-items: center;
            font-size: 12px;
            font-weight: 700;
            color: #0f172a;
        }}
        .chatbox-avatar.assistant {{
            background: linear-gradient(135deg, #f97316, #f59e0b);
            box-shadow: 0 8px 18px rgba(249, 115, 22, 0.32);
        }}
        .chatbox-avatar.user {{
            background: linear-gradient(135deg, #60a5fa, #38bdf8);
            box-shadow: 0 8px 18px rgba(56, 189, 248, 0.32);
        }}
        .chatbox-bubble {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 12px;
            line-height: 1.5;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.38);
            max-width: 82%;
            word-break: break-word;
            white-space: pre-wrap;
        }}
        .chatbox-bubble.user {{
            margin-left: auto;
        }}
        .chatbox-bubble.assistant {{
            margin-right: auto;
        }}
        .chatbox-input {{
            display: flex;
            gap: 8px;
            padding: 12px 14px 14px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(6, 8, 15, 0.85);
        }}
        .chatbox-input input {{
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            padding: 10px 12px;
            border-radius: 10px;
            font-size: 12px;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .chatbox-input input:focus {{
            border-color: rgba(249, 115, 22, 0.7);
            box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.24);
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
            box-shadow: 0 12px 28px rgba(249, 115, 22, 0.4);
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        }}
        .chatbox-input button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 14px 32px rgba(249, 115, 22, 0.5);
        }}
        .chatbox-input button:active {{
            transform: translateY(0);
            box-shadow: 0 8px 18px rgba(249, 115, 22, 0.36);
        }}
        /* Streamlit form overrides scoped to this component */
        form[aria-label="chatbox_form"],
        [data-chatbox-id="{component_id}"] [data-testid="stForm"] {{
            margin: 0;
        }}
        form[aria-label="chatbox_form"] > div,
        [data-chatbox-id="{component_id}"] [data-testid="stForm"] > div {{
            padding: 0 !important;
        }}
        form[aria-label="chatbox_form"] [data-testid="stTextInput"] > div > div,
        [data-chatbox-id="{component_id}"] [data-testid="stForm"] [data-testid="stTextInput"] > div > div {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        form[aria-label="chatbox_form"] [data-testid="stTextInput"] input,
        [data-chatbox-id="{component_id}"] [data-testid="stForm"] [data-testid="stTextInput"] input {{
            background: transparent;
            color: #e2e8f0;
            padding: 10px 12px;
            font-size: 12px;
        }}
        form[aria-label="chatbox_form"] [data-testid="stTextInput"] input:focus,
        [data-chatbox-id="{component_id}"] [data-testid="stForm"] [data-testid="stTextInput"] input:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        form[aria-label="chatbox_form"] [data-testid="stTextInput"] > div:hover,
        [data-chatbox-id="{component_id}"] [data-testid="stForm"] [data-testid="stTextInput"] > div:hover {{
            border-color: rgba(249, 115, 22, 0.7);
            box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.18);
        }}
        form[aria-label="chatbox_form"] [data-testid="baseButton-secondaryFormSubmit"],
        [data-chatbox-id="{component_id}"] [data-testid="baseButton-secondaryFormSubmit"] {{
            background: linear-gradient(135deg, #f97316, #f59e0b) !important;
            color: #0f172a !important;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            height: 42px;
            box-shadow: 0 12px 28px rgba(249, 115, 22, 0.42);
        }}
        form[aria-label="chatbox_form"] [data-testid="baseButton-secondaryFormSubmit"]:hover,
        [data-chatbox-id="{component_id}"] [data-testid="baseButton-secondaryFormSubmit"]:hover {{
            filter: brightness(1.03);
            transform: translateY(-1px);
            box-shadow: 0 14px 32px rgba(249, 115, 22, 0.5);
        }}
        form[aria-label="chatbox_form"] [data-testid="baseButton-secondaryFormSubmit"]:active,
        [data-chatbox-id="{component_id}"] [data-testid="baseButton-secondaryFormSubmit"]:active {{
            transform: translateY(0);
            box-shadow: 0 10px 22px rgba(249, 115, 22, 0.36);
        }}
        @media (max-width: 640px) {{
            .chatbox-wrapper {{
                width: 100%;
                max-width: none;
            }}
            .chatbox-messages {{
                max-height: 420px;
                min-height: 240px;
            }}
        }}
        .st-key-chatbox_input {{
          text-wrap: wrap;
        }}
        </style>
<div class="chatbox-wrapper" data-chatbox-id="{component_id}">
<div class="chatbox">
<div class="chatbox-header">
<div class="chatbox-dot"></div>
<div class="chatbox-title">Chat Assistant</div>
<div class="chatbox-subtitle">Powered by Gemini 3.0 Pro</div>
</div>
<div class="chatbox-messages">
{messages_html}
</div>
        """,
        unsafe_allow_html=True,
    )

    with chat_container.form("chatbox_form", clear_on_submit=True, border=False):
        cols = st.columns([5, 1.4])
        with cols[0]:
            prompt = st.text_input(
                "Message",
                key="chatbox_input",
                label_visibility="collapsed",
                placeholder="Type a message and press Enter...",
            )
        with cols[1]:
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and prompt.strip():
            text = prompt.strip()
            _append_message("user", text)
            _append_message("assistant", f"I'll echo: {text}")
            st.rerun()

    chat_container.markdown(
        """
</div>
</div>
        """,
        unsafe_allow_html=True,
    )