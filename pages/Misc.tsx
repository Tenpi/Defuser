import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
MiscTabContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
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
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedMiscTab = localStorage.getItem("miscTab")
        if (savedMiscTab) setMiscTab(savedMiscTab)
    }, [])

    useEffect(() => {
        localStorage.setItem("miscTab", String(miscTab))
    }, [miscTab])

    const miscTabsJSX = () => {
        return (
            <div className="train-tab-row">
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
        if (miscTab === "simplify sketch") {
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