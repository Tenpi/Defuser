import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
ClassifyTabContext, ClassifyFolderLocationContext, EpochsContext, SaveStepsContext,
GradientAccumulationStepsContext, LearningRateContext, ResolutionContext, LearningFunctionContext,
LearningRateTEContext, FolderLocationContext, TrainNameContext, PreviewStepsContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
import TrainClassifier from "./TrainClassifier"
import AIDetector from "./AIDetector"
import ModelConvert from "./ModelConvert"
import ConvertEmbedding from "./ConvertEmbedding"
import TrainUnconditional from "./TrainUnconditional"
import "./styles/generate.less"

const Classify: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {classifyTab, setClassifyTab} = useContext(ClassifyTabContext)
    const {folderLocation, setFolderLocation} = useContext(FolderLocationContext)
    const {classifyFolderLocation, setClassifyFolderLocation} = useContext(ClassifyFolderLocationContext)
    const {epochs, setEpochs} = useContext(EpochsContext)
    const {saveSteps, setSaveSteps} = useContext(SaveStepsContext)
    const {previewSteps, setPreviewSteps} = useContext(PreviewStepsContext)
    const {learningRate, setLearningRate} = useContext(LearningRateContext)
    const {gradientAccumulationSteps, setGradientAccumulationSteps} = useContext(GradientAccumulationStepsContext)
    const {resolution, setResolution} = useContext(ResolutionContext)
    const {learningFunction, setLearningFunction} = useContext(LearningFunctionContext)
    const {learningRateTE, setLearningRateTE} = useContext(LearningRateTEContext)
    const {trainName, setTrainName} = useContext(TrainNameContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedFolderLocation = localStorage.getItem("folderLocation")
        if (savedFolderLocation) setFolderLocation(savedFolderLocation)
        const savedClassifyFolderLocation = localStorage.getItem("classifyFolderLocation")
        if (savedClassifyFolderLocation) setClassifyFolderLocation(savedClassifyFolderLocation)
        const savedClassifyTab = localStorage.getItem("classifyTab")
        if (savedClassifyTab) setClassifyTab(savedClassifyTab)
        const savedEpochs = localStorage.getItem("epochs")
        if (savedEpochs) setEpochs(savedEpochs)
        const savedSaveSteps = localStorage.getItem("saveSteps")
        if (savedSaveSteps) setSaveSteps(savedSaveSteps)
        const savedPreviewSteps = localStorage.getItem("previewSteps")
        if (savedPreviewSteps) setPreviewSteps(savedPreviewSteps)
        const savedLearningRate = localStorage.getItem("learningRate")
        if (savedLearningRate) setLearningRate(savedLearningRate)
        const savedGradientAccumulationSteps = localStorage.getItem("gradientAccumulationSteps")
        if (savedGradientAccumulationSteps) setGradientAccumulationSteps(savedGradientAccumulationSteps)
        const savedResolution = localStorage.getItem("resolution")
        if (savedResolution) setResolution(savedResolution)
        const savedLearningFunction = localStorage.getItem("learningFunction")
        if (savedLearningFunction) setLearningFunction(savedLearningFunction)
        const savedLearningRateTE = localStorage.getItem("learningRateTE")
        if (savedLearningRateTE) setLearningRateTE(savedLearningRateTE)
        const savedTrainName = localStorage.getItem("trainName")
        if (savedTrainName) setTrainName(savedTrainName)
    }, [])

    useEffect(() => {
        localStorage.setItem("classifyFolderLocation", String(classifyFolderLocation))
        localStorage.setItem("classifyTab", String(classifyTab))
        localStorage.setItem("epochs", String(epochs))
        localStorage.setItem("saveSteps", String(saveSteps))
        localStorage.setItem("previewSteps", String(previewSteps))
        localStorage.setItem("learningRate", String(learningRate))
        localStorage.setItem("gradientAccumulationSteps", String(gradientAccumulationSteps))
        localStorage.setItem("resolution", String(resolution))
        localStorage.setItem("learningFunction", String(learningFunction))
        localStorage.setItem("learningRateTE", String(learningRateTE))
    }, [classifyFolderLocation, classifyTab, epochs, saveSteps, previewSteps, learningRate, 
        gradientAccumulationSteps, learningFunction, resolution, learningRateTE])

    const classifyTabsJSX = () => {
        return (
            <div className="train-tab-row">
                <div className="train-tab-container" onClick={() => setClassifyTab("ai detector")}>
                    <span className={classifyTab === "ai detector" ? "train-tab-text-selected" : "train-tab-text"}>AI Detector</span>
                </div>
                <div className="train-tab-container" onClick={() => setClassifyTab("train classifier")}>
                    <span className={classifyTab === "train classifier" ? "train-tab-text-selected" : "train-tab-text"}>Train Classifier</span>
                </div>
                <div className="train-tab-container" onClick={() => setClassifyTab("model convert")}>
                    <span className={classifyTab === "model convert" ? "train-tab-text-selected" : "train-tab-text"}>Model Convert</span>
                </div>
                <div className="train-tab-container" onClick={() => setClassifyTab("convert embedding")}>
                    <span className={classifyTab === "convert embedding" ? "train-tab-text-selected" : "train-tab-text"}>Convert Embedding</span>
                </div>
                <div className="train-tab-container" onClick={() => setClassifyTab("unconditional")}>
                    <span className={classifyTab === "unconditional" ? "train-tab-text-selected" : "train-tab-text"}>Unconditional</span>
                </div>
            </div>
        )
    }

    const getTab = () => {
        if (classifyTab === "train classifier") {
            return <TrainClassifier/>
        } else if (classifyTab === "ai detector") {
            return <AIDetector/>
        } else if (classifyTab === "model convert") {
            return <ModelConvert/>
        } else if (classifyTab === "convert embedding") {
            return <ConvertEmbedding/>
        } else if (classifyTab === "unconditional") {
            return <TrainUnconditional/>
        } 
    }

    return (
        <div className="generate" onMouseEnter={() => setEnableDrag(false)}>
            <CheckpointBar/>
            {classifyTabsJSX()}
            {getTab()}
        </div>
    )
}

export default Classify