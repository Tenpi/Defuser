import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
ImageBrightnessContext, ImageContrastContext, ImageHueContext, ImageSaturationContext, SocketContext} from "../Context"
import functions from "../structures/Functions"
import pkg from "../package.json"
import Draggable from "react-draggable"
import "./styles/update.less"
import axios from "axios"

const UpdateDialog: React.FunctionComponent = () => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {imageHue, setImageHue} = useContext(ImageHueContext)
    const {imageSaturation, setImageSaturation} = useContext(ImageSaturationContext)
    const {socket, setSocket} = useContext(SocketContext)
    const [newVersion, setNewVersion] = useState("")
    const [visible, setVisible] = useState(false)
    const [toggleShow, setToggleShow] = useState(false)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    useEffect(() => {
        if (!socket) return
        const updateAvailable = async (data: any) => {
            if (data.version) {
                setNewVersion(data.version)
                setVisible(true)
                setTimeout(() => {
                    setToggleShow(true)
                }, 200)
            }
        }
        socket.on("update available", updateAvailable)
        return () => {
            socket.off("update available", updateAvailable)
        }
    }, [socket])

    useEffect(() => {
        if (!visible) setToggleShow(false)
    }, [visible])

    const update = async () => {
        window.open(`${pkg.repository.url}/releases`, "_blank")?.focus()
        axios.post("/dismiss-update")
        setVisible(false)
    }

    const dismiss = async () => {
        axios.post("/dismiss-update")
        setVisible(false)
    }

    if (visible) {
        return (
            <Draggable handle=".update-dialog-text-container">
                <div className={`update-dialog ${toggleShow ? "update-dialog-show" : ""}`} onMouseEnter={() => setEnableDrag(false)}>
                    <div className="update-dialog-row">
                        <div className="update-dialog-text-container">
                            <span className="update-dialog-text">A new version<span className="update-dialog-text-bold">{newVersion}</span>
                            is available. Make sure that you update.</span>
                        </div>
                    </div>
                    <div className="update-dialog-row">
                        <button className="update-dialog-row-button-reject" onClick={dismiss}>Dismiss</button>
                        <button className="update-dialog-row-button-accept" onClick={update}>Update</button>
                    </div>
                </div>
            </Draggable>
        )
    }
    return null
}

export default UpdateDialog