import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
SavedPromptsContext, SavedPromptsNovelAIContext, SavedPromptsHolaraAIContext, GeneratorContext} from "../Context"
import functions from "../structures/Functions"
import CheckpointBar from "../components/CheckpointBar"
import "./styles/generate.less"
import axios from "axios"

let timeout = null as any

const SavedPrompts: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {savedPrompts, setSavedPrompts} = useContext(SavedPromptsContext)
    const {savedPromptsNovelAI, setSavedPromptsNovelAI} = useContext(SavedPromptsNovelAIContext)
    const {savedPromptsHolaraAI, setSavedPromptsHolaraAI} = useContext(SavedPromptsHolaraAIContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedPrompts = localStorage.getItem("savedPrompts")
        if (savedPrompts) setSavedPrompts(JSON.parse(savedPrompts))
        const savedPromptsNovelAI = localStorage.getItem("savedPrompts-novel-ai")
        if (savedPromptsNovelAI) setSavedPromptsNovelAI(JSON.parse(savedPromptsNovelAI))
        const savedPromptsHolaraAI = localStorage.getItem("savedPrompts-holara-ai")
        if (savedPromptsHolaraAI) setSavedPromptsHolaraAI(JSON.parse(savedPromptsHolaraAI))
    }, [])

    useEffect(() => {
        localStorage.setItem("savedPrompts", JSON.stringify(savedPrompts))
        localStorage.setItem("savedPrompts-novel-ai", JSON.stringify(savedPromptsNovelAI))
        localStorage.setItem("savedPrompts-holara-ai", JSON.stringify(savedPromptsHolaraAI))
    }, [savedPrompts, savedPromptsNovelAI, savedPromptsHolaraAI])

    const changeSavedPrompts = (event: any) => {
        if (timeout) clearTimeout(timeout)
        const prompts = event.target.value.split("\n")
        if (generator === "novel ai") {
            setSavedPromptsNovelAI(prompts)
        } else if (generator === "holara ai") {
            setSavedPromptsHolaraAI(prompts)
        } else {
            setSavedPrompts(prompts)
        }
        timeout = setTimeout(() => {
            axios.post("/save-prompts", {prompts, generator})
        }, 1000)
    }

    const getSavedPrompts = () => {
        if (generator === "novel ai") return savedPromptsNovelAI.join("\n")
        if (generator === "holara ai") return savedPromptsHolaraAI.join("\n")
        return savedPrompts.join("\n")
    }

    return (
        <div className="generate" onMouseEnter={() => setEnableDrag(false)}>
            <CheckpointBar/>
            <div className="settings-bar">
                <div className="settings-bar-row">
                    <span className="settings-bar-text">Saved Prompts:</span>
                    <textarea className="settings-bar-textarea" style={{minHeight: "75vh", width: "70vw"}} spellCheck={false} value={getSavedPrompts()} onChange={(event) => changeSavedPrompts(event)}></textarea>
                </div>
            </div>
        </div>
    )
}

export default SavedPrompts