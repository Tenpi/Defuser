import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
TrainTabContext, FolderLocationContext, EpochsContext, SaveEpochsContext, PreviewEpochsContext, PreviewPromptContext,
GradientAccumulationStepsContext, LearningRateContext, ResolutionContext, LearningFunctionContext, TrainNameContext,
LearningRateTEContext, PreviewStepsContext, SaveStepsContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
import TrainTag from "./TrainTag"
import TrainSource from "./TrainSource"
import TrainTextualInversion from "./TrainTextualInversion"
import TrainHypernetwork from "./TrainHypernetwork"
import TrainLora from "./TrainLora"
import TrainDreamBooth from "./TrainDreamBooth"
import TrainCheckpoint from "./TrainCheckpoint"
import TrainMerge from "./TrainMerge"
import TrainCrop from "./TrainCrop"
import "./styles/generate.less"

const Train: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {trainTab, setTrainTab} = useContext(TrainTabContext)
    const {folderLocation, setFolderLocation} = useContext(FolderLocationContext)
    const {epochs, setEpochs} = useContext(EpochsContext)
    const {saveEpochs, setSaveEpochs} = useContext(SaveEpochsContext)
    const {saveSteps, setSaveSteps} = useContext(SaveStepsContext)
    const {previewEpochs, setPreviewEpochs} = useContext(PreviewEpochsContext)
    const {previewSteps, setPreviewSteps} = useContext(PreviewStepsContext)
    const {previewPrompt, setPreviewPrompt} = useContext(PreviewPromptContext)
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
        const savedTrainTab = localStorage.getItem("trainTab")
        if (savedTrainTab) setTrainTab(savedTrainTab)
        const savedEpochs = localStorage.getItem("epochs")
        if (savedEpochs) setEpochs(savedEpochs)
        const savedSaveEpochs = localStorage.getItem("saveEpochs")
        if (savedSaveEpochs) setSaveEpochs(savedSaveEpochs)
        const savedPreviewEpochs = localStorage.getItem("previewEpochs")
        if (savedPreviewEpochs) setPreviewEpochs(savedPreviewEpochs)
        const savedPreviewPrompt = localStorage.getItem("previewPrompt")
        if (savedPreviewPrompt) setPreviewPrompt(savedPreviewPrompt)
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
        const savedSteps = localStorage.getItem("saveSteps")
        if (savedSteps) setSaveSteps(savedSteps)
        const savedPreviewSteps = localStorage.getItem("previewSteps")
        if (savedPreviewSteps) setPreviewSteps(savedPreviewSteps)
    }, [])

    useEffect(() => {
        localStorage.setItem("folderLocation", String(folderLocation))
        localStorage.setItem("trainTab", String(trainTab))
        localStorage.setItem("epochs", String(epochs))
        localStorage.setItem("saveEpochs", String(saveEpochs))
        localStorage.setItem("previewEpochs", String(previewEpochs))
        localStorage.setItem("previewPrompt", String(previewPrompt))
        localStorage.setItem("learningRate", String(learningRate))
        localStorage.setItem("gradientAccumulationSteps", String(gradientAccumulationSteps))
        localStorage.setItem("resolution", String(resolution))
        localStorage.setItem("learningFunction", String(learningFunction))
        localStorage.setItem("learningRateTE", String(learningRateTE))
        localStorage.setItem("trainName", String(trainName))
        localStorage.setItem("saveSteps", String(saveSteps))
        localStorage.setItem("previewSteps", String(previewSteps))
    }, [folderLocation, trainTab, epochs, saveEpochs, previewEpochs, previewPrompt, learningRate, 
        gradientAccumulationSteps, learningFunction, resolution, learningRateTE, trainName,
        saveSteps, previewSteps])

    const trainTabsJSX = () => {
        return (
            <div className="train-tab-row">
                <div className="train-tab-container" onClick={() => setTrainTab("crop")}>
                    <span className={trainTab === "crop" ? "train-tab-text-selected" : "train-tab-text"}>Crop</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("tag")}>
                    <span className={trainTab === "tag" ? "train-tab-text-selected" : "train-tab-text"}>Tag</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("source")}>
                    <span className={trainTab === "source" ? "train-tab-text-selected" : "train-tab-text"}>Source</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("textual inversion")}>
                    <span className={trainTab === "textual inversion" ? "train-tab-text-selected" : "train-tab-text"}>Textual Inversion</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("hypernetwork")}>
                    <span className={trainTab === "hypernetwork" ? "train-tab-text-selected" : "train-tab-text"}>Hypernetwork</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("lora")}>
                    <span className={trainTab === "lora" ? "train-tab-text-selected" : "train-tab-text"}>LoRA</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("dreambooth")}>
                    <span className={trainTab === "dreambooth" ? "train-tab-text-selected" : "train-tab-text"}>DreamBooth</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("checkpoint")}>
                    <span className={trainTab === "checkpoint" ? "train-tab-text-selected" : "train-tab-text"}>Checkpoint</span>
                </div>
                <div className="train-tab-container" onClick={() => setTrainTab("merge")}>
                    <span className={trainTab === "merge" ? "train-tab-text-selected" : "train-tab-text"}>Merge</span>
                </div>
            </div>
        )
    }

    const getTab = () => {
        if (trainTab === "crop") {
            return <TrainCrop/>
        } else if (trainTab === "tag") {
            return <TrainTag/>
        } else if (trainTab === "source") {
            return <TrainSource/>
        } else if (trainTab === "textual inversion") {
            return <TrainTextualInversion/>
        } else if (trainTab === "hypernetwork") {
            return <TrainHypernetwork/>
        } else if (trainTab === "lora") {
            return <TrainLora/>
        } else if (trainTab === "dreambooth") {
            return <TrainDreamBooth/>
        } else if (trainTab === "checkpoint") {
            return <TrainCheckpoint/>
        } else if (trainTab === "merge") {
            return <TrainMerge/>
        }
    }

    return (
        <div className="generate" onMouseEnter={() => setEnableDrag(false)}>
            <CheckpointBar/>
            {trainTabsJSX()}
            {getTab()}
        </div>
    )
}

export default Train