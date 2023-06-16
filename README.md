## Agent AWS
Agent AWS is an automated, AI-powered agent that uses HuggingFace Transformers paired with numerous different foundation models.

An “agent” here is a large language model (LLM) equipped with a prompt and access to a specific set of tools. 
Tools are self-contained functions that perform a singular task that the agent can use when given an instruction.

Agent AWS contains a set of tools that allows it to query AWS documentation, generating code, and creating architectural diagrams.

## QuickStart

Here is how you can easily get started using the Agent

Checkout the code
```
git clone https://github.com/build-on-aws/building-gen-ai-agent-on-aws.git
cd building-gen-ai-agent-on-aws
```

Run the Agent
```
pip install -r requirements.txt
# Learn how to get your own free key here https://huggingface.co/docs/hub/security-tokens
export HUGGING_FACE_KEY=YOUR_KEY
streamlit run agent_aws_st.py
# View at localhost:8501
```

## Example Outputs
Here are some example outputs

<div align="center"><img src="images/code_example.png" alt="Writing Code"></div>
<div align="center"><img src="images/diagram_example.png" alt="Creating Diagram"></div>
<div align="center"><img src="images/query_example.png" alt="Query example"></div>

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the [LICENSE file](LICENSE.md).

## Code of Conduct

We want the Learner Library to be useful to everyone and welcome PRs and Issues. We expect those who use this repo to abide by our [Code of Conduct](https://aws.github.io/code-of-conduct).
