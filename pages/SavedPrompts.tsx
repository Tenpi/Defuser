import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, SavedPromptsContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
import "./styles/generate.less"

const SavedPrompts: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {savedPrompts, setSavedPrompts} = useContext(SavedPromptsContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedPrompts = localStorage.getItem("savedPrompts")
        if (savedPrompts) setSavedPrompts(JSON.parse(savedPrompts))
    }, [])

    useEffect(() => {
        localStorage.setItem("savedPrompts", JSON.stringify(savedPrompts))
    }, [savedPrompts])

    return (
        <div className="generate" onMouseEnter={() => setEnableDrag(false)}>
            <CheckpointBar/>
            <div className="settings-bar">
                <div className="settings-bar-row">
                    <span className="settings-bar-text">Saved Prompts:</span>
                    <textarea className="settings-bar-textarea" style={{minHeight: "75vh", width: "70vw"}} spellCheck={false} value={savedPrompts.join("\n")} onChange={(event) => setSavedPrompts(event.target.value.split("\n"))}></textarea>
                </div>
            </div>
        </div>
    )
}

export default SavedPrompts