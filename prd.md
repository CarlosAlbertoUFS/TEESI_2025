# **Project Requirements Document: Chat Bot Enem **

The following table outlines the detailed functional requirements of Chat Bot Enem.

| Requirement ID | Description | User Story | Expected Behavior/Outcome |
|:--------------:|-------------|------------|---------------------------|
|RN001           | Submission documents | As a student, I want Chat to answer ENEM questions based on the texts of documents that I provide.| The Chat application should provide a button for submitting documents. The submitted documents should be loaded and transformed into embeddings and then stored in a vector database in memory, to then be queried by a retriever and chained to a prompt that will be sent along with the user's input.|

