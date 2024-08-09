# AiDea
An AiDea about your portfolio.

## Overview

AiDea is a comprehensive project aimed at managing and analyzing your portfolio using advanced financial models. The project is designed to evolve over time, with initial design and implementation focused on creating a robust foundation that will be extended in further stages.

## Project Structure

### 1. Project Design
The **Project Design.pdf** document outlines the initial blueprint of the project, providing a detailed roadmap for future development. This design will be iteratively refined as the project progresses.

### 2. backend.ipynb
The **backend.ipynb** notebook serves as the core of the project's backend, including:

- **Flask App**: A web application framework to handle requests from the frontend.
- **SQLAlchemy**: An ORM (Object-Relational Mapping) tool for interacting with the database, ensuring seamless data management.
- **Model Section**: This part of the notebook is dedicated to developing and testing financial models. These models are used to perform various analyses and take actions as requested by the frontend of the web app.

### 3. Templates Directory
The **Templates** directory contains the HTML pages for the web application. Each file in this directory is an HTML document that defines the structure and content of the web pages, including any necessary scripts to enable interactive features.

### 4. Instances
The **Instances** directory contains the database, which includes the tables configured for the project. These tables are:

- **User**
- **TrainedModels**
- **Symbols**
- **Temporary Password**

The number of tables can be extended, and the existing ones can be modified in future stages.


## Future Development

In the subsequent stages of development, the backend.ipynb will be modularized to improve maintainability and scalability. This modularization will involve splitting the notebook into distinct components for:

- **Web App**: Handling the frontend-backend interactions.
- **Database**: Managing the data storage and retrieval processes.
- **Models**: Isolating the financial models for easier updates and testing.
- **Backend**: Streamlining the core backend functionalities.

This approach will ensure that each component of AiDea can be independently developed, tested, and extended, making the project more robust and easier to maintain.

---

This README will be updated as the project evolves, reflecting the latest changes and enhancements.



