import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, 
SiteLightnessContext, SocketContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import "./styles/traintag.less"
import axios from "axios"

const ConvertEmbedding: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {socket, setSocket} = useContext(SocketContext)
    const [mode, setMode] = useState("file")
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedMode = localStorage.getItem("convertEmbeddingMode")
        if (savedMode) setMode(savedMode)
    }, [])

    useEffect(() => {
        localStorage.setItem("convertEmbeddingMode", String(mode))
    }, [mode])

    const convert = async () => {
        const json = {} as any
        json.mode = mode
        await axios.post("/convert-embedding", json)
    }

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-column">
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Embedding Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onClick={convert}>
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>
                    </label>
                </div>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <button className="shade-sketch-button" style={{backgroundColor: mode === "file" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setMode("file")}>File</button>
                        <button className="shade-sketch-button" style={{backgroundColor: mode === "folder" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setMode("folder")}>Folder</button>
                    </div>
                </div>
                <span className="train-tag-settings-title">Convert an SD1.5 embedding to SDXL.</span>
            </div>
        </div>
    )
}

export default ConvertEmbedding