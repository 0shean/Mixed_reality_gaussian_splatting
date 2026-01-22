# Interactive Language-Queryable Gaussian Scenes

Repo by Sergejs Zahovskis, Dmitry Knorre, James Conant, and Yael Fassbind for the 2025 Mixed Reality course at ETH.

## Overview

Our objective is to make a VR app that allows users to interact with a photorealistic 3D scene with natural language queries.
They can say an object out loud then our application will highlight it.
Simple, but complicated to implement on the VR headset.
This repo is a fork of this great project [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting?tab=readme-ov-file).
All the code here except for the Python server runs on the headset.
It is divided into three main components.

### Application (Assets/)

Most of the interaction side of our application lives here.
There are different Unity objects for responding to voice input, manipulating buttons and text, and moving the camera around.
We wrote some custom scripts to send input to the backend.

### Backend Python server (occam_backend/)

The Python server we wrote to perform CLIP similarity scores lives here.
It's a simple `occam_server.py` that has a few endpoints for computing a big buffer of relevancy scores over every Gaussian in the scene and sending it back in a streaming fashion.
There are a few helpers for reading in `.ply` files and loading the models.
We run this in the background during the operation of the VR app, which connects to it over the network.

## Gaussian Splat rendering package (package/)

This is built off the skeleton from the repo we forked.
It's a local Unity package that overrides the camera's default rendering pipeline with a custom ground-up implementation of Gaussian splatting.
The rendering is implemented via a C# driver program (`GaussianSplatRenderer.cs`) that invokes different shaders on the GPU to sort the Gaussians, cull invisible ones, and alpha blend them.
We tinkered heavily with this package to add the new rendering mode for highlighting language features.
