import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
MiscTabContext, ClassifyFolderLocationContext, EpochsContext, SaveStepsContext,
GradientAccumulationStepsContext, LearningRateContext, ResolutionContext, LearningFunctionContext,
LearningRateTEContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
import TrainClassifier from "./TrainClassifier"
import AIDetector from "./AIDetector"
import SimplifySketch from "./SimplifySketch"
import ShadeSketch from "./ShadeSketch"
import ColorizeSketch from "./ColorizeSketch"
import LayerDivide from "./LayerDivide"
import "./styles/generate.less"

const Misc: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {miscTab, setMiscTab} = useContext(MiscTabContext)
    const {classifyFolderLocation, setClassifyFolderLocation} = useContext(ClassifyFolderLocationContext)
    const {epochs, setEpochs} = useContext(EpochsContext)
    const {saveSteps, setSaveSteps} = useContext(SaveStepsContext)
    const {learningRate, setLearningRate} = useContext(LearningRateContext)
    const {gradientAccumulationSteps, setGradientAccumulationSteps} = useContext(GradientAccumulationStepsContext)
    const {resolution, setResolution} = useContext(ResolutionContext)
    const {learningFunction, setLearningFunction} = useContext(LearningFunctionContext)
    const {learningRateTE, setLearningRateTE} = useContext(LearningRateTEContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedClassifyFolderLocation = localStorage.getItem("classifyFolderLocation")
        if (savedClassifyFolderLocation) setClassifyFolderLocation(savedClassifyFolderLocation)
        const savedMiscTab = localStorage.getItem("miscTab")
        if (savedMiscTab) setMiscTab(savedMiscTab)
        const savedEpochs = localStorage.getItem("epochs")
        if (savedEpochs) setEpochs(savedEpochs)
        const savedSaveSteps = localStorage.getItem("saveSteps")
        if (savedSaveSteps) setSaveSteps(savedSaveSteps)
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
    }, [])

    useEffect(() => {
        localStorage.setItem("classifyFolderLocation", String(classifyFolderLocation))
        localStorage.setItem("miscTab", String(miscTab))
        localStorage.setItem("epochs", String(epochs))
        localStorage.setItem("saveSteps", String(saveSteps))
        localStorage.setItem("learningRate", String(learningRate))
        localStorage.setItem("gradientAccumulationSteps", String(gradientAccumulationSteps))
        localStorage.setItem("resolution", String(resolution))
        localStorage.setItem("learningFunction", String(learningFunction))
        localStorage.setItem("learningRateTE", String(learningRateTE))
    }, [classifyFolderLocation, miscTab, epochs, saveSteps, learningRate, 
        gradientAccumulationSteps, learningFunction, resolution, learningRateTE])

    const miscTabsJSX = () => {
        return (
            <div className="train-tab-row">
                <div className="train-tab-container" onClick={() => setMiscTab("ai detector")}>
                    <span className={miscTab === "ai detector" ? "train-tab-text-selected" : "train-tab-text"}>AI Detector</span>
                </div>
                <div className="train-tab-container" onClick={() => setMiscTab("train classifier")}>
                    <span className={miscTab === "train classifier" ? "train-tab-text-selected" : "train-tab-text"}>Train Classifier</span>
                </div>
                <div className="train-tab-container" onClick={() => setMiscTab("simplify sketch")}>
                    <span className={miscTab === "simplify sketch" ? "train-tab-text-selected" : "train-tab-text"}>Simplify Sketch</span>
                </div>
                <div className="train-tab-container" onClick={() => setMiscTab("shade sketch")}>
                    <span className={miscTab === "shade sketch" ? "train-tab-text-selected" : "train-tab-text"}>Shade Sketch</span>
                </div>
                <div className="train-tab-container" onClick={() => setMiscTab("colorize sketch")}>
                    <span className={miscTab === "colorize sketch" ? "train-tab-text-selected" : "train-tab-text"}>Colorize Sketch</span>
                </div>
                <div className="train-tab-container" onClick={() => setMiscTab("layer divide")}>
                    <span className={miscTab === "layer divide" ? "train-tab-text-selected" : "train-tab-text"}>Layer Divide</span>
                </div>
            </div>
        )
    }

    const getTab = () => {
        if (miscTab === "train classifier") {
            return <TrainClassifier/>
        } else if (miscTab === "ai detector") {
            return <AIDetector/>
        } else if (miscTab === "simplify sketch") {
            return <SimplifySketch/>
        } else if (miscTab === "shade sketch") {
            return <ShadeSketch/>
        } else if (miscTab === "colorize sketch") {
            return <ColorizeSketch/>
        } else if (miscTab === "layer divide") {
            return <LayerDivide/>
        }
    }

    return (
        <div className="generate" onMouseEnter={() => setEnableDrag(false)}>
            <CheckpointBar/>
            {miscTabsJSX()}
            {getTab()}
        </div>
    )
}

export default Misc