# Import libraries
import PIL
import streamlit as st

import agent_setup


@st.cache_resource
def load_agent():
    """
    Load the agent

    Returns:
        agent (HFagent): The agent object

    """
    return agent_setup.start_agent()


def sidebar() -> None:
    """
    Purpose:
        Shows the side bar
    Args:
        N/A
    Returns:
        N/A
    """

    st.sidebar.image(
        "https://d1.awsstatic.com/gamedev/Programs/OnRamp/gt-well-architected.4234ac16be6435d0ddd4ca693ea08106bc33de9f.png",
        use_column_width=True,
    )

    st.sidebar.markdown(
        "Agent AWS is an automated, AI-powered agent that uses HuggingFace Transformers paired with numerous different foundation models"
    )


def app() -> None:
    """
    Purpose:
        Controls the app flow
    Args:
        N/A
    Returns:
        N/A
    """

    # Spin up the sidebar
    sidebar()

    agent = load_agent()

    query = st.text_input("Query:")

    if st.button("Submit Query"):
        with st.spinner("Generating..."):
            answer = agent.run(query)

            if type(answer) == PIL.PngImagePlugin.PngImageFile:
                st.image(answer)
            elif type(answer) == dict:
                st.markdown(answer["ans"])
                docs = answer["docs"].split("\n")

                with st.expander("Resources"):
                    for doc in docs:
                        st.write(doc)
            else:
                st.code(answer)


def main() -> None:
    """
    Purpose:
        Controls the flow of the streamlit app
    Args:
        N/A
    Returns:
        N/A
    """

    # Start the streamlit app
    st.title("Agent AWS")
    st.subheader("Ask and Learn")

    app()


if __name__ == "__main__":
    main()
